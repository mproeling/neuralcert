"""Reusable PyTorch channel models for numerical discovery.

Only model representation belongs here; objective construction and exact
certification are kept in their respective packages.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


def inv_softplus(r: np.ndarray) -> np.ndarray:
    """softplus^{-1}(r) = log(e^r - 1), computed stably.

    The naive log(expm1(r)) overflows to +inf for r >= 710 and poisons every
    parameter with NaN — which is exactly what happened for k >= 355 with
    rate_cap = 2k. For large r, log(e^r - 1) = r + log1p(-e^{-r}).
    """
    r = np.asarray(r, dtype=np.float64)
    out = np.empty_like(r)
    small = r < 30.0
    out[small] = np.log(np.expm1(np.clip(r[small], 1e-12, None)))
    out[~small] = r[~small] + np.log1p(-np.exp(-r[~small]))
    return out


def channel_parts(g: nn.Module, x: torch.Tensor):
    if hasattr(g, "parts"):
        return g.parts(x)
    v = g(x)
    return v, torch.zeros(v.shape[-1], dtype=v.dtype, device=v.device)


def channel_log_parts(g: nn.Module, x: torch.Tensor):
    """(log raw, rates) for sign-definite channels: g_j = e^{log_raw_j} e^{-rho_j x}.

    Required by the log-space recursion. A module that cannot guarantee
    raw > 0 must not implement this."""
    if not hasattr(g, "log_parts"):
        raise TypeError(
            f"{type(g).__name__} does not expose log_parts(); the log-space "
            f"recursion needs sign-definite channels. Use --channel-sign free "
            f"(valid only for small k) or a positive channel model.")
    return g.log_parts(x)


class FixedDictionary(nn.Module):
    """Baseline channels g(x) = x^p e^{-r x} — the classical GPY-type
    subspace, for orientation. Exposes parts() so the exponential is
    factored out here too."""

    def __init__(self, k: int, powers=(0, 1, 2), n_rates: int = 4) -> None:
        super().__init__()
        rates = [0.0] + list(np.geomspace(max(k / 2.0, 1.0), 2.0 * k, n_rates - 1))
        self.spec = [(p, r) for p in powers for r in rates]
        self.m = len(self.spec)

    def parts(self, x: torch.Tensor):
        cols = [x.pow(p) if p > 0 else torch.ones_like(x) for (p, _) in self.spec]
        raw = torch.stack(cols, dim=1)
        rates = torch.tensor([r for (_, r) in self.spec],
                             dtype=x.dtype, device=x.device)
        return raw, rates

    def log_parts(self, x: torch.Tensor):
        lx = x.clamp_min(1e-300).log()
        cols = [p * lx if p > 0 else torch.zeros_like(x) for (p, _) in self.spec]
        rates = torch.tensor([r for (_, r) in self.spec], dtype=x.dtype,
                             device=x.device)
        return torch.stack(cols, dim=1).clamp_min(-700.0), rates

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw, rates = self.parts(x)
        return raw * torch.exp(-x.unsqueeze(-1) * rates.unsqueeze(0))


class LinearX(nn.Module):
    """g(x) = x: the second preflight control. Unlike g == 1 (whose psi is
    constant, making the positive linear stencil EXACT at any N), this has
    xi(s) = 2p log s + const — genuinely non-constant — so it exercises the
    interpolation. Closed form: h = x^2 has Laplace transform 2/s^3, so
    nu_p(s) = 2^p s^{3p-1} / Gamma(3p); with H(y) = y^3/3, G(y) = y^2/2 both
    A and B reduce to Beta functions and

        R = k (3/4) B(3k-3, 5) / B(3k-3, 4) = 3k / (3k + 1).
    """

    def parts(self, x: torch.Tensor):
        return x.unsqueeze(-1), torch.zeros(1, dtype=x.dtype, device=x.device)

    def log_parts(self, x: torch.Tensor):
        return x.clamp_min(1e-300).log().unsqueeze(-1), \
            torch.zeros(1, dtype=x.dtype, device=x.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1)


class ConstantOne(nn.Module):
    """g == 1: the closed-form control. rates = 0, so v9 reduces exactly to
    the pure algebraic factorisation here."""

    def parts(self, x: torch.Tensor):
        return torch.ones_like(x).unsqueeze(-1), torch.zeros(1, dtype=x.dtype,
                                                             device=x.device)

    def log_parts(self, x: torch.Tensor):
        return torch.zeros_like(x).unsqueeze(-1), torch.zeros(1, dtype=x.dtype,
                                                              device=x.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(x).unsqueeze(-1)


class ValidationError(RuntimeError):
    """A computed quantity violated a property the exact problem guarantees.
    These are not warnings: if they fire, the discretised pencil no longer
    represents the variational problem and every number downstream is void."""
