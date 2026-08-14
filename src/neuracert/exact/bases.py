"""Exact polynomial bases with stable float evaluation.

Every basis here supplies TWO representations of the same object, and the
separation is not cosmetic:

  * `integer_coeffs(n)` -- exact integer (or integer/scale) expansion in the
    monomial basis, consumed by the EXACT engine.  Bigints cannot cancel, so
    the monomial form is harmless there.
  * `design(x, d)` -- float evaluation via the three-term recurrence,
    consumed by the FLOAT engine.  Never evaluate the monomial expansion in
    floats: at degree >~ 60 the shifted-Legendre monomial coefficients reach
    ~8^d and np.polyval loses every significant digit to cancellation, while
    |P~_n| <= 1 on [0,1] at any degree.

A basis is a good fit for this pipeline when (a) it is orthogonal, so the
least-squares projection is well conditioned, (b) its coefficients are
integers or integers over a common power of a small base, so the exact
engine never leaves Z, and (c) it has a three-term recurrence, so float
evaluation is stable and O(d).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from functools import lru_cache

import numpy as np


class ExactBasis(ABC):
    """A polynomial basis with exact integer coefficients."""

    name: str = "unnamed"

    @abstractmethod
    def integer_coeffs(self, n: int) -> tuple[list[int], int]:
        """(monomial coefficients, common denominator) of basis element n."""

    @abstractmethod
    def design(self, x: np.ndarray, d: int) -> np.ndarray:
        """(len(x), d+1) float design matrix, evaluated stably."""

    def integer_coeff_table(self, d: int) -> list[list[int]]:
        """Monomial coefficients of elements 0..d on a COMMON denominator.

        Returns numerators only; `common_denominator(d)` gives the shared
        denominator.  Putting every element on one denominator is what lets
        the exact engine treat a projected channel as a single integer
        polynomial over a single scale.
        """
        den = self.common_denominator(d)
        table = []
        for n in range(d + 1):
            coeffs, dn = self.integer_coeffs(n)
            mult = den // dn
            table.append([c * mult for c in coeffs] + [0] * (d - n))
        return table

    def common_denominator(self, d: int) -> int:
        den = 1
        for n in range(d + 1):
            den = math.lcm(den, self.integer_coeffs(n)[1])
        return den


# ---------------------------------------------------------------------------
# Shifted Legendre on [0, 1]  -- the Maynard basis
# ---------------------------------------------------------------------------
class ShiftedLegendre(ExactBasis):
    r"""P~_n(x) = P_n(2x - 1) on [0, 1], sup-norm 1, integer coefficients.

        P~_n(x) = sum_j (-1)^{n+j} C(n,j) C(n+j,j) x^j
        (n+1) P~_{n+1} = (2n+1)(2x-1) P~_n - n P~_{n-1}

    Antiderivative identity used by the certifiers:
        int_0^y P~_n = (P~_{n+1}(y) - P~_{n-1}(y)) / (2(2n+1))   (n >= 1)
    the constant vanishing because P~_n(0) = (-1)^n.
    """

    name = "shifted_legendre"

    @lru_cache(maxsize=None)
    def integer_coeffs(self, n: int) -> tuple[tuple[int, ...], int]:
        return tuple((-1) ** (n + j) * math.comb(n, j) * math.comb(n + j, j)
                     for j in range(n + 1)), 1

    def design(self, x: np.ndarray, d: int) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        out = np.empty((x.size, d + 1), dtype=np.float64)
        out[:, 0] = 1.0
        if d >= 1:
            out[:, 1] = 2.0 * x - 1.0
        z = 2.0 * x - 1.0
        for n in range(1, d):
            out[:, n + 1] = ((2 * n + 1) * z * out[:, n]
                             - n * out[:, n - 1]) / (n + 1)
        return out

    def antiderivative_design(self, y: np.ndarray, d: int) -> np.ndarray:
        """(len(y), d+1) matrix of int_0^y P~_n, exact identity, stable."""
        y = np.asarray(y, dtype=np.float64)
        phi = self.design(y, d + 1)
        out = np.empty((y.size, d + 1), dtype=np.float64)
        out[:, 0] = y
        for n in range(1, d + 1):
            out[:, n] = (phi[:, n + 1] - phi[:, n - 1]) / (2 * (2 * n + 1))
        return out


# ---------------------------------------------------------------------------
# Krawtchouk  -- the Delsarte basis
# ---------------------------------------------------------------------------
class Krawtchouk(ExactBasis):
    r"""K_i(x; n, q), the eigenvalues of the Hamming association scheme.

        K_i(x) = sum_j (-1)^j (q-1)^{i-j} C(x,j) C(n-x, i-j)

    Integer valued at integer x, with the recurrence (binary q = 2 shown;
    the general-q form is implemented)

        (i+1) K_{i+1}(x) = (q-1)(n-i) K_i(x) - ... - (n-i+1)(q-1) K_{i-1}(x)

    For Delsarte duals the object that matters is the VALUE TABLE K_i(j) for
    integer i, j in [0, n] -- exactly an integer matrix -- so the "basis" is
    used through `value_table` rather than through monomial coefficients.
    Feasibility of a dual polynomial is then a finite set of integer sign
    checks with no analysis anywhere.
    """

    name = "krawtchouk"

    def __init__(self, n: int, q: int = 2) -> None:
        if n < 0 or q < 2:
            raise ValueError("need n >= 0, q >= 2")
        self.n, self.q = int(n), int(q)

    def value(self, i: int, x: int) -> int:
        """K_i(x) as an exact integer."""
        n, q = self.n, self.q
        return sum((-1) ** j * (q - 1) ** (i - j) * math.comb(x, j)
                   * math.comb(n - x, i - j)
                   for j in range(0, i + 1) if i - j <= n - x)

    @lru_cache(maxsize=None)
    def value_table(self, d: int | None = None) -> tuple[tuple[int, ...], ...]:
        """K[i][x] for i in 0..d, x in 0..n; exact integers."""
        d = self.n if d is None else int(d)
        return tuple(tuple(self.value(i, x) for x in range(self.n + 1))
                     for i in range(d + 1))

    def integer_coeffs(self, i: int) -> tuple[tuple[int, ...], int]:
        """Monomial coefficients in x, on a common denominator i!.

        Obtained by expanding the falling-factorial products exactly; used
        only when a continuous relaxation is wanted.  The discrete Delsarte
        machinery should use `value_table` instead.
        """
        n, q = self.n, self.q
        num = [0] * (i + 1)
        for j in range(i + 1):
            # C(x, j) = x(x-1)...(x-j+1)/j!  and C(n-x, i-j) likewise
            pa = _falling_poly(j, 0)                      # numerator of C(x,j)
            pb = _falling_poly(i - j, n, negate=True)     # C(n-x, i-j)
            prod = _polymul(pa, pb)
            scale = math.factorial(i) // (math.factorial(j)
                                          * math.factorial(i - j))
            w = (-1) ** j * (q - 1) ** (i - j) * scale
            for deg, c in enumerate(prod):
                num[deg] += w * c
        return tuple(num), math.factorial(i)

    def design(self, x: np.ndarray, d: int) -> np.ndarray:
        """Float design matrix via the three-term recurrence in i."""
        x = np.asarray(x, dtype=np.float64)
        n, q = float(self.n), float(self.q)
        out = np.empty((x.size, d + 1), dtype=np.float64)
        out[:, 0] = 1.0
        if d >= 1:
            out[:, 1] = (q - 1) * n - q * x
        for i in range(1, d):
            a = (q - 1) * (n - i) + i - q * x
            out[:, i + 1] = (a * out[:, i]
                             - (q - 1) * (n - i + 1) * out[:, i - 1]) / (i + 1)
        return out


def _falling_poly(j: int, shift: int, negate: bool = False) -> list[int]:
    """Coefficients of prod_{r=0}^{j-1} (s - r) where s = x or s = shift - x."""
    poly = [1]
    for r in range(j):
        if negate:                       # (shift - x) - r  ->  (shift-r) - x
            factor = [shift - r, -1]
        else:                            # x - r
            factor = [-r, 1]
        poly = _polymul(poly, factor)
    return poly


def _polymul(a: list[int], b: list[int]) -> list[int]:
    out = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        if ai:
            for j, bj in enumerate(b):
                out[i + j] += ai * bj
    return out


# ---------------------------------------------------------------------------
# Gegenbauer / ultraspherical  -- the spherical-code and energy basis
# ---------------------------------------------------------------------------
class Gegenbauer(ExactBasis):
    r"""C_n^{(lam)}(t) on [-1, 1], normalised to C_n(1) = 1.

    With lam = (d-2)/2 these are the zonal spherical harmonics of S^{d-1};
    positive-definiteness of a kernel sum_n f_n C_n is EXACTLY f_n >= 0, so
    the Delsarte/Yudin bound machinery for codes, packings and energy all
    run through this basis.  Coefficients are rational with denominators
    built from the Pochhammer recurrence, so the exact engine works over a
    common denominator returned alongside the numerators.

        (n+1) C_{n+1} = ((2n + d - 2) t C_n - (n + d - 3) C_{n-1}) / (n+d-3+1)
    (the normalised-at-1 form; see `design` for the implemented recurrence)
    """

    name = "gegenbauer"

    def __init__(self, dim: int) -> None:
        if dim < 2:
            raise ValueError("need dim >= 2")
        self.dim = int(dim)

    @lru_cache(maxsize=None)
    def integer_coeffs(self, n: int) -> tuple[tuple[int, ...], int]:
        from fractions import Fraction
        d = self.dim
        prev: list[Fraction] = [Fraction(1)]
        cur: list[Fraction] = [Fraction(0), Fraction(1)]
        if n == 0:
            return (1,), 1
        if n == 1:
            return (0, 1), 1
        for m in range(1, n):
            a = Fraction(2 * m + d - 2, m + d - 2)
            b = Fraction(m, m + d - 2)
            nxt = [Fraction(0)] * (m + 2)
            for i, c in enumerate(cur):
                nxt[i + 1] += a * c
            for i, c in enumerate(prev):
                nxt[i] -= b * c
            prev, cur = cur, nxt
        den = 1
        for c in cur:
            den = math.lcm(den, c.denominator)
        return tuple(int(c * den) for c in cur), den

    def design(self, t: np.ndarray, d: int) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        dim = self.dim
        out = np.empty((t.size, d + 1), dtype=np.float64)
        out[:, 0] = 1.0
        if d >= 1:
            out[:, 1] = t
        for m in range(1, d):
            a = (2 * m + dim - 2) / (m + dim - 2)
            b = m / (m + dim - 2)
            out[:, m + 1] = a * t * out[:, m] - b * out[:, m - 1]
        return out


BASES = {
    "shifted_legendre": ShiftedLegendre,
    "krawtchouk": Krawtchouk,
    "gegenbauer": Gegenbauer,
}
