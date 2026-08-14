"""Float discovery output -> exact dyadic rational channels.

This is the bridge between the two halves of the pipeline, and it is where
soundness is either preserved or quietly lost.  The invariant:

    NOTHING downstream may depend on the projection being ACCURATE.

The projected channels are exact rational objects.  Whatever they are, the
exact engine computes the true Rayleigh quotient OF THEM, so the certified
bound is valid however badly the projection approximates the neural
channels.  Projection error costs bound QUALITY, never bound VALIDITY.  The
residual diagnostics returned here are therefore a quality report, not a
correctness gate -- but a large residual means the exact bound will be
worse than the float one, so it is worth watching.

The dyadic denominator is shared by all channels, which lets the exact
engine carry one power-of-two scale instead of an LCM of many.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bases import ExactBasis


@dataclass(frozen=True)
class ProjectedChannels:
    """Exact channels on a common dyadic denominator 2**bits."""

    basis_coeffs: list[list[int]]     # numerators in the ORTHOGONAL basis
    monomial_coeffs: list[list[int]]  # numerators in the MONOMIAL basis
    bits: int                         # denominator = 2**bits (times basis den)
    basis_denominator: int            # extra common denominator of the basis
    degree: int
    l2_residual: list[float]          # relative, per channel
    linf_residual: list[float]        # relative, per channel

    @property
    def denominator(self) -> int:
        return (1 << self.bits) * self.basis_denominator

    def report(self) -> str:
        return (f"projection: degree {self.degree}, {len(self.basis_coeffs)} "
                f"channels, denominator 2^{self.bits}"
                + (f" * {self.basis_denominator}"
                   if self.basis_denominator != 1 else "")
                + f", max rel L2 {max(self.l2_residual):.2e}, "
                f"max rel Linf {max(self.linf_residual):.2e}")


def project_to_dyadic(x_fine: np.ndarray, g_fine: np.ndarray, basis: ExactBasis,
                      degree: int, bits: int) -> ProjectedChannels:
    """Least-squares projection onto `basis`, rounded to dyadic rationals.

    x_fine    : (N,) sample points
    g_fine    : (N, m) channel values
    basis     : an ExactBasis; its design matrix must be STABLE at `degree`
    bits      : dyadic precision; coefficients become round(c * 2**bits)

    Both representations are returned.  The exact engine consumes the
    monomial numerators (bigints cannot cancel).  The float engine must NOT:
    at degree >~ 60 the monomial coefficients of an orthogonal basis reach
    ~8^d and polyval loses every digit, so float work goes through the
    recurrence on `basis_coeffs`.
    """
    g_fine = np.atleast_2d(np.asarray(g_fine, dtype=np.float64))
    if g_fine.shape[0] != x_fine.size:
        g_fine = g_fine.T
    m = g_fine.shape[1]

    phi = basis.design(np.asarray(x_fine, dtype=np.float64), degree)
    coef, *_ = np.linalg.lstsq(phi, g_fine, rcond=None)

    table = basis.integer_coeff_table(degree)
    basis_den = basis.common_denominator(degree)

    basis_coeffs, mono_coeffs, l2r, lir = [], [], [], []
    unit = 1 << bits
    for j in range(m):
        bn = [int(round(float(coef[i, j]) * unit)) for i in range(degree + 1)]
        mono = [0] * (degree + 1)
        for i, ci in enumerate(bn):
            if ci:
                for deg, cf in enumerate(table[i]):
                    if cf:
                        mono[deg] += ci * cf
        basis_coeffs.append(bn)
        mono_coeffs.append(mono)

        approx = phi @ (np.asarray(bn, dtype=np.float64) / unit)
        err = approx - g_fine[:, j]
        s2 = float(np.sqrt(np.mean(g_fine[:, j] ** 2))) or 1.0
        si = float(np.max(np.abs(g_fine[:, j]))) or 1.0
        l2r.append(float(np.sqrt(np.mean(err ** 2))) / s2)
        lir.append(float(np.max(np.abs(err))) / si)

    return ProjectedChannels(basis_coeffs=basis_coeffs,
                             monomial_coeffs=mono_coeffs, bits=bits,
                             basis_denominator=basis_den, degree=degree,
                             l2_residual=l2r, linf_residual=lir)


def scale_by_log_diagonal(values: np.ndarray, log_diag: np.ndarray,
                          exponent: float, label: str = "vector"
                          ) -> np.ndarray:
    """values_j * exp(exponent * log_diag_j), without overflow.

    Normalised by one common positive scalar so max|out| = 1; a global
    rescaling leaves every Rayleigh quotient unchanged.  Zeros stay exactly
    zero; non-finite inputs are REJECTED rather than silently becoming NaN.
    """
    v = np.asarray(values, dtype=np.float64)
    ld = np.asarray(log_diag, dtype=np.float64)
    if v.shape != ld.shape:
        raise ValueError(f"{label}: shape mismatch {v.shape} != {ld.shape}")
    if np.any(~np.isfinite(ld)):
        bad = np.where(~np.isfinite(ld))[0].tolist()
        raise ValueError(f"{label}: non-finite log diagonal at {bad}")
    if np.any(~np.isfinite(v)):
        bad = np.where(~np.isfinite(v))[0].tolist()
        raise ValueError(f"{label}: non-finite coefficients at {bad}")
    out = np.zeros_like(v)
    nz = v != 0.0
    if not np.any(nz):
        raise ValueError(f"{label}: all coefficients are zero")
    logabs = np.log(np.abs(v[nz])) + exponent * ld[nz]
    out[nz] = np.sign(v[nz]) * np.exp(logabs - float(np.max(logabs)))
    if not np.all(np.isfinite(out)) or np.max(np.abs(out)) == 0.0:
        raise FloatingPointError(f"{label}: scale-safe conversion failed")
    return out


def dyadic_rebalance(monomial_coeffs: list[list[int]], active: list[int],
                     log_diag: np.ndarray, power: int, base_bits: int,
                     scale_bits: int = 48, verbose: bool = True
                     ) -> tuple[list[list[int]], int]:
    """Exactly rescale active channels by dyadic approximations of a
    diagonal preconditioner, preserving a single shared denominator.

    The float engine works in the preconditioned frame Fhat_j =
    exp(-logA_jj/2) F_j.  For a channel entering the k-dimensional object as
    a `power`-fold tensor product, that change of basis moves onto the
    one-dimensional factor as exp(-logA_jj/(2*power)).  We quantise each
    positive scale to an exact dyadic a_j / 2**scale_bits.  A COMMON
    logarithmic offset is removed first: it multiplies every basis vector by
    the same scalar and so cannot change a Rayleigh quotient.
    """
    active = list(active)
    ld = np.asarray(log_diag, dtype=np.float64)[np.asarray(active, dtype=int)]
    if np.any(~np.isfinite(ld)):
        bad = np.where(~np.isfinite(ld))[0].tolist()
        raise ValueError(f"rebalance: non-finite log diagonal at {bad}")
    root_logs = -0.5 * ld / float(power)
    root_logs -= float(np.max(root_logs))       # scales in (0, 1]
    scales = np.exp(root_logs)
    unit = 1 << scale_bits
    nums = [max(1, int(v)) for v in np.rint(scales * unit)]
    scaled = [[int(c) * a for c in monomial_coeffs[idx]]
              for idx, a in zip(active, nums)]
    if verbose:
        rel = max(abs(a / unit - s) / s for a, s in zip(nums, scales))
        print(f"  dyadic rebalance: scales in [{min(scales):.3e}, "
              f"{max(scales):.3e}], max rel. quantisation {rel:.2e}")
    return scaled, base_bits + scale_bits
