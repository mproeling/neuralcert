"""Numerical primitives used by discovery pipelines.

This module contains floating-point quadrature, interpolation, device selection
and convolution-chain planning.  It deliberately has no dependency on the
exact-arithmetic :mod:`maynard_tools.certification` package.
"""
from __future__ import annotations
import math
import numpy as np
import scipy.linalg
from scipy.special import gammaln
import torch


def resolve_device(spec: str) -> torch.device:
    spec = spec.lower()
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    dev = torch.device(spec)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    if dev.type == "mps" and not (hasattr(torch.backends, "mps")
                                  and torch.backends.mps.is_available()):
        raise RuntimeError("MPS requested but this PyTorch build/device lacks it")
    return dev


def parse_torch_dtype(name: str) -> torch.dtype:
    table = {"float64": torch.float64, "fp64": torch.float64, "double": torch.float64,
             "float32": torch.float32, "fp32": torch.float32, "single": torch.float32}
    try:
        return table[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype {name!r}") from exc


def gauss_legendre_01(n: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(n)
    return 0.5 * (nodes + 1.0), 0.5 * weights


def cheb_lobatto_01(N: int) -> np.ndarray:
    """Chebyshev-Lobatto nodes on [0,1], increasing, INCLUDING both endpoints.
    Endpoint inclusion is the point: every convolution target s_i u_r lies in
    [0, s_i] and is interpolated, never extrapolated."""
    if N < 3:
        raise ValueError("representation grid needs N >= 3")
    i = np.arange(N)
    return 0.5 * (1.0 - np.cos(np.pi * i / (N - 1)))


def cheb_lobatto_bary_weights(N: int) -> np.ndarray:
    """Closed-form barycentric weights for Chebyshev-Lobatto: (-1)^i, halved
    at the endpoints. Lebesgue constant O(log N)."""
    w = np.ones(N)
    w[1::2] = -1.0
    w[0] *= 0.5
    w[-1] *= 0.5
    return w


def clenshaw_curtis_01(N: int) -> np.ndarray:
    """Clenshaw-Curtis weights on the Chebyshev-Lobatto nodes of [0,1].
    Used only for the channel L2 normalisation, so no extra point set."""
    n = N - 1
    j = np.arange(1, n // 2 + 1)
    b = np.where(2 * j == n, 1.0, 2.0) / (4.0 * j * j - 1.0)
    i = np.arange(N)
    cosmat = np.cos(2.0 * np.outer(i, j) * np.pi / n)
    c = np.where((i == 0) | (i == n), 1.0, 2.0)
    return 0.5 * (c / n) * (1.0 - cosmat @ b)


def gauss_jacobi_01(nq: int, p: float, q: float) -> tuple[np.ndarray, np.ndarray]:
    """Nodes u_r and NORMALISED weights what_r (sum = 1) with
        int_0^1 f(u) u^{p-1}(1-u)^{q-1} du = B(p,q) sum_r what_r f(u_r).
    scipy uses weight (1-x)^alpha (1+x)^beta on [-1,1]; with u = (1+x)/2 that
    is alpha = q-1, beta = p-1.

    Constructed via Golub-Welsch (eigenvalues of the Jacobi tridiagonal),
    NOT scipy.roots_jacobi: the latter computes the zeroth moment
    mu0 = 2^(a+b+1) B(a+1,b+1) in LINEAR space, which overflows float64 once
    the channel power p exceeds ~170 (k >~ 170), returning NaN weights and
    crashing the g==1 preflight at large k.  Golub-Welsch needs only the
    recurrence coefficients (no Gamma/Beta evaluation), and because we return
    weights normalised to sum 1, the overflowing mu0 scale cancels
    identically -- the B(p,q) factor is applied separately downstream in
    log-space (see log_beta / step['logB']).  Validated against roots_jacobi
    to ~1e-14 for p <= 60; finite and correct at p = 2500."""
    a = q - 1.0
    b = p - 1.0
    n = int(nq)
    ab = a + b
    k = np.arange(n, dtype=np.float64)
    # monic Jacobi recurrence -> symmetric tridiagonal diagonal
    denom0 = (2.0 * k + ab) * (2.0 * k + ab + 2.0)
    diag = np.divide(b * b - a * a, denom0,
                     out=np.zeros_like(k), where=denom0 != 0.0)
    diag[0] = (b - a) / (ab + 2.0) if (ab + 2.0) != 0.0 else 0.0
    # off-diagonal (sqrt of monic beta_k), k = 1 .. n-1
    if n > 1:
        kk = np.arange(1, n, dtype=np.float64)
        num = 4.0 * kk * (kk + a) * (kk + b) * (kk + ab)
        den = (2.0 * kk + ab) ** 2 * (2.0 * kk + ab + 1.0) * (2.0 * kk + ab - 1.0)
        off = np.sqrt(np.divide(num, den, out=np.zeros_like(kk),
                                where=den != 0.0))
        T = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
    else:
        T = np.diag(diag)
    ev, V = np.linalg.eigh(T)
    w = V[0, :] ** 2
    w = w / w.sum()                          # mu0 cancels -> exact sum 1
    u = 0.5 * (ev + 1.0)                      # [-1,1] -> [0,1]
    idx = np.argsort(u)
    return u[idx], w[idx]


def log_beta(p: float, q: float) -> float:
    return float(gammaln(p) + gammaln(q) - gammaln(p + q))


def barycentric_matrix(nodes: np.ndarray, wb: np.ndarray,
                       targets: np.ndarray) -> np.ndarray:
    nodes = np.asarray(nodes, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    D = targets[:, None] - nodes[None, :]
    exact = np.abs(D) < 1e-14
    W = wb[None, :] / np.where(exact, 1.0, D)
    M = W / W.sum(axis=1, keepdims=True)
    hit = exact.any(axis=1)
    if hit.any():
        M[hit, :] = 0.0
        r, c = np.where(exact)
        M[r, c] = 1.0
    return M


def torch_barycentric(nodes: torch.Tensor, wb: torch.Tensor,
                      targets: torch.Tensor) -> torch.Tensor:
    """Batched barycentric interpolation matrix. targets: (..., Q) ->
    (..., Q, N). Used for the per-pair outer windows, whose node positions
    change every iteration (they follow the channel rates)."""
    D = targets.unsqueeze(-1) - nodes
    exact = D.abs() < 1e-14
    W = wb / torch.where(exact, torch.ones_like(D), D)
    M = W / W.sum(dim=-1, keepdim=True)
    if bool(exact.any()):
        hit = exact.any(dim=-1, keepdim=True)
        M = torch.where(hit, exact.to(M.dtype), M)
    return M


def linear_interp_pairs(nodes: np.ndarray, targets: np.ndarray):
    """Piecewise-linear interpolation as (i0, i1, w0, w1) with w0, w1 >= 0,
    w0 + w1 = 1. Requires targets inside [nodes[0], nodes[-1]] — guaranteed
    here because the representation grid is endpoint-inclusive and every
    convolution target s_i u_r lies in [0, s_i] subset [0, L].

    Nonnegativity is not a numerical nicety, it is what keeps A a Gram matrix.
    The discrete entry is A_{jl} = sum_alpha c_alpha u_alpha(j) u_alpha(l)
    with c_alpha a product of quadrature weights (positive) and interpolation
    weights; if any interpolation weight is negative, c_alpha can be negative
    and A is the Gram matrix of nothing. Barycentric/spectral weights are
    signed, which is why the spectral path produces a genuinely indefinite A.
    """
    nodes = np.asarray(nodes, float)
    t = np.clip(np.asarray(targets, float), nodes[0], nodes[-1])
    j = np.clip(np.searchsorted(nodes, t, side="right") - 1, 0, nodes.size - 2)
    x0, x1 = nodes[j], nodes[j + 1]
    w1 = (t - x0) / (x1 - x0)
    w1 = np.clip(w1, 0.0, 1.0)
    return j, j + 1, 1.0 - w1, w1


def addition_chain(p: int) -> list[tuple[str, int, int]]:
    """Left-to-right binary powering chain: ('sq',a,a) squares the current
    accumulator, ('mul',a,1) multiplies it by the base. ~2 log2(p) entries."""
    if p < 1:
        raise ValueError("p must be >= 1")
    ops: list[tuple[str, int, int]] = []
    cur = 1
    for bit in bin(p)[3:]:
        ops.append(("sq", cur, cur))
        cur *= 2
        if bit == "1":
            ops.append(("mul", cur, 1))
            cur += 1
    assert cur == p, (p, cur)
    return ops


def sequential_chain(p: int) -> list[tuple[str, int, int]]:
    """Reference chain: p-1 multiplies by the base, same psi machinery."""
    return [("mul", a, 1) for a in range(1, p)]
