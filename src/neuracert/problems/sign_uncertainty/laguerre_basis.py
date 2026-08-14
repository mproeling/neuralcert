"""Problem-level exact Laguerre basis definitions shared across stages."""

from __future__ import annotations

import math
from fractions import Fraction


def lag_coeffs_frac(n: int, d: int) -> list[Fraction]:
    """Ascending coefficients of L_n^((d-2)/2)(2u), exactly over Q."""
    alpha = Fraction(d - 2, 2)
    coefficients = []
    for j in range(n + 1):
        binomial = Fraction(1)
        for t in range(n - j):
            binomial *= (alpha + j + 1 + t) / Fraction(t + 1)
        coefficients.append(binomial * Fraction((-2) ** j, math.factorial(j)))
    return coefficients


def orders_for(n_basis: int, sign: int) -> list[int]:
    """Laguerre orders in the requested Fourier eigenspace."""
    if n_basis < 1:
        raise ValueError("n_basis must be positive")
    if sign not in {-1, 1}:
        raise ValueError("sign must be -1 or +1")
    start = 0 if sign == 1 else 1
    return [start + 2 * index for index in range(n_basis)]
