"""Association schemes: the algebraic data every Delsarte bound needs.

A symmetric association scheme on a finite set X with d classes is fixed,
for LP purposes, by four exact objects:

    |X|          the ground set size
    v_i          valencies         (i = 0..d)
    m_j          multiplicities    (j = 0..d)
    P_{ji}       first eigenmatrix: the eigenvalue of the adjacency matrix
                 A_i on the j-th eigenspace

Everything else is derived.  In particular the SECOND eigenmatrix, which is
what the Delsarte dual is actually expressed in, follows from

    v_i Q_{ij} = m_j P_{ji}                                            (*)

so a new scheme is added by supplying valencies, multiplicities and P --
three tables of exact integers -- and nothing else.

WHY (*) IS THE RIGHT PLACE TO PUT THE WORK
Because it comes with a free, complete correctness test.  The eigenmatrices
of any symmetric association scheme satisfy

    P Q = Q P = |X| I

as exact matrices.  `check_orthogonality` verifies this over Q with no
tolerance.  A mistyped Eberlein or Krawtchouk formula, a wrong multiplicity,
a transposed index -- all of them break it.  This is the single highest-value
test in the whole Delsarte stack, because the failure mode it catches
(plausible-looking bounds that are silently below the true optimum) is
otherwise invisible.

THE DELSARTE BOUND IN THIS LANGUAGE
For a code C with inner distribution a, MacWilliams gives (aQ)_j >= 0 and
(aQ)_0 = |C|.  So if F(i) = sum_j f_j Q_{ij} has f_0 > 0, f_j >= 0 for
j >= 1, and F(i) <= 0 for every class i on which a_i may be nonzero, then

    f_0 |C| <= sum_i a_i F(i) <= F(0) = sum_j f_j m_j
    ==>  |C| <= (sum_j f_j m_j) / f_0.

Note the sign condition lives on the ACHIEVABLE classes, not the excluded
ones.  Getting that backwards is the classic error; see `problems.delsarte`.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from fractions import Fraction
from functools import lru_cache


class AssociationScheme(ABC):
    """Symmetric association scheme, exact arithmetic throughout."""

    name: str = "unnamed"

    # -- required of an implementation ------------------------------------
    @property
    @abstractmethod
    def n_classes(self) -> int:
        """d, so classes are 0..d."""

    @property
    @abstractmethod
    def size(self) -> int:
        """|X|."""

    @abstractmethod
    def valencies(self) -> tuple[int, ...]:
        """v_i, i = 0..d.  v_0 = 1."""

    @abstractmethod
    def multiplicities(self) -> tuple[int, ...]:
        """m_j, j = 0..d.  m_0 = 1."""

    @abstractmethod
    def eigenvalue(self, i: int, j: int) -> Fraction:
        """P_{ji} = P_i(j): eigenvalue of A_i on eigenspace j."""

    # -- derived ----------------------------------------------------------
    @lru_cache(maxsize=None)
    def dual_eigenvalue(self, i: int, j: int) -> Fraction:
        """Q_{ij} = m_j P_{ji} / v_i, from v_i Q_{ij} = m_j P_{ji}."""
        return (Fraction(self.multiplicities()[j])
                * Fraction(self.eigenvalue(i, j))
                / Fraction(self.valencies()[i]))

    def P_matrix(self) -> list[list[Fraction]]:
        """P[j][i] = P_i(j), cached.

        The default fill calls `eigenvalue` (d+1)^2 times, and each of those
        is itself an O(d) alternating sum -- cubic, which is what made the
        structural pass the bottleneck rather than the matrix product.
        Subclasses with a three-term recurrence override `_fill_P` to build
        the whole table in O(d^2).
        """
        cached = getattr(self, "_P_cache", None)
        if cached is None:
            cached = self._fill_P()
            self._P_cache = cached
        return cached

    def _fill_P(self) -> list[list[Fraction]]:
        d = self.n_classes
        return [[Fraction(self.eigenvalue(i, j)) for i in range(d + 1)]
                for j in range(d + 1)]

    def Q_matrix(self) -> list[list[Fraction]]:
        d = self.n_classes
        return [[self.dual_eigenvalue(i, j) for j in range(d + 1)]
                for i in range(d + 1)]

    def check_structure(self) -> tuple[bool, str]:
        """O(d^2) identities that need no matrix product.

        Cheap, and between them they pin down a great deal:

            v_0 = m_0 = 1,  sum_i v_i = sum_j m_j = |X|
            P_i(0) = v_i    (A_i acts on the trivial eigenspace by its valency)
            P_0(j) = 1      (A_0 = I)
            |P_i(j)| <= v_i (an adjacency eigenvalue cannot exceed the valency)
            sum_j Q_ij = |X| delta_i0   (the k=0 column of Q P = |X| I)

        The last two are real orthogonality consequences, not bookkeeping, so
        this is a meaningful gate on its own when the full O(d^3) product is
        out of reach.
        """
        d, N = self.n_classes, self.size
        vs, ms = self.valencies(), self.multiplicities()
        if vs[0] != 1 or ms[0] != 1:
            return False, f"v_0 = {vs[0]}, m_0 = {ms[0]}; both must be 1"
        if sum(vs) != N:
            return False, f"sum of valencies {sum(vs)} != |X| = {N}"
        if sum(ms) != N:
            return False, f"sum of multiplicities {sum(ms)} != |X| = {N}"
        P = self.P_matrix()
        for i in range(d + 1):
            if P[0][i] != vs[i]:
                return False, f"P_{i}(0) = {P[0][i]} != v_{i}"
        for j in range(d + 1):
            if P[j][0] != 1:
                return False, f"P_0({j}) != 1 (A_0 must be the identity)"
            for i in range(d + 1):
                if abs(P[j][i]) > vs[i]:
                    return False, (f"|P_{i}({j})| = {abs(P[j][i])} exceeds the "
                                   f"valency v_{i} = {vs[i]}")
        # sum_j Q_ij = (sum_j m_j P_ji) / v_i, so one integer sum per row
        for i in range(d + 1):
            acc = sum(Fraction(ms[j]) * P[j][i] for j in range(d + 1))
            want = Fraction(N * vs[i]) if i == 0 else Fraction(0)
            if acc != want:
                return False, (f"sum_j Q_{i}j = {acc / vs[i]}, expected "
                               f"{want / vs[i]}")
        return True, "structural identities hold"

    def check_orthogonality(self) -> tuple[bool, str]:
        """P Q = |X| I, exactly.  Returns (ok, message)."""
        d, N = self.n_classes, self.size
        P, Q = self.P_matrix(), self.Q_matrix()
        for j in range(d + 1):
            for kk in range(d + 1):
                s = sum(P[j][i] * Q[i][kk] for i in range(d + 1))
                want = Fraction(N) if j == kk else Fraction(0)
                if s != want:
                    return False, (f"(PQ)[{j}][{kk}] = {s}, expected {want}")
        vs, ms = self.valencies(), self.multiplicities()
        if vs[0] != 1 or ms[0] != 1:
            return False, f"v_0 = {vs[0]}, m_0 = {ms[0]}; both must be 1"
        if sum(vs) != N:
            return False, f"sum of valencies {sum(vs)} != |X| = {N}"
        if sum(ms) != N:
            return False, f"sum of multiplicities {sum(ms)} != |X| = {N}"
        for i in range(d + 1):
            if self.eigenvalue(i, 0) != vs[i]:
                return False, f"P_{i}(0) = {self.eigenvalue(i, 0)} != v_{i}"
            if self.dual_eigenvalue(0, i) != ms[i]:
                return False, f"Q_0({i}) != m_{i}"
        return True, "P Q = |X| I; valencies, multiplicities and row 0 consistent"

    def check_orthogonality_modular(self, n_primes: int = 6
                                    ) -> tuple[bool, str]:
        """P Q = |X| I on the FULL grid, checked modulo several primes.

        Exact rational arithmetic on the matrix product costs O(d^3)
        operations on integers with hundreds of digits, which is hopeless
        past d ~ 300.  Reducing mod p turns every entry into a machine word
        and lets numpy do the product.

        Primes are chosen near 2^26 so that p^2 (d+1) stays inside int64 and
        the accumulation cannot overflow.

        This is a probabilistic check and is labelled as such: a corrupted
        entry escapes only if it is divisible by every prime used.  With six
        independent primes that is far below the probability of a hardware
        fault, and it covers the WHOLE grid rather than the sample that the
        exact subgrid check can afford.
        """
        try:
            import numpy as np
        except ImportError:  # pragma: no cover
            return self.check_orthogonality_subgrid()

        d, N = self.n_classes, self.size
        P = self.P_matrix()
        vs, ms = self.valencies(), self.multiplicities()
        if (d + 1) * (1 << 52) >= (1 << 63):  # pragma: no cover
            return False, "scheme too large for the int64 modular check"

        Pi = [[int(P[j][i]) for i in range(d + 1)] for j in range(d + 1)]
        primes, cand = [], (1 << 26) - 1
        while len(primes) < n_primes:
            if _is_small_prime(cand):
                primes.append(cand)
            cand -= 2

        for p in primes:
            if any(v % p == 0 for v in vs):
                continue                      # p divides a valency; skip it
            Pm = np.array([[c % p for c in row] for row in Pi], dtype=np.int64)
            inv_v = [pow(int(v) % p, p - 2, p) for v in vs]
            Qm = np.array([[(ms[k] % p) * Pm[k][i] % p * inv_v[i] % p
                            for k in range(d + 1)] for i in range(d + 1)],
                          dtype=np.int64)
            prod = (Pm @ Qm) % p
            want = np.zeros_like(prod)
            np.fill_diagonal(want, N % p)
            if not np.array_equal(prod, want):
                j, k = [int(v[0]) for v in np.nonzero(prod != want)]
                return False, (f"(PQ)[{j}][{k}] != |X| delta mod {p}")
        return True, (f"P Q = |X| I on the full {d + 1}x{d + 1} grid modulo "
                      f"{len(primes)} primes, plus structural identities")

    def check_orthogonality_subgrid(self, samples: int = 12
                                    ) -> tuple[bool, str]:
        """P Q = |X| I on a sampled subgrid of eigenspaces, exactly."""
        d, N = self.n_classes, self.size
        idx = sorted(set(list(range(min(samples, d + 1)))
                         + list(range(max(0, d + 1 - samples), d + 1))
                         + list(range(0, d + 1, max(1, (d + 1) // samples)))))
        P, Q = self.P_matrix(), self.Q_matrix()
        for j in idx:
            for kk in idx:
                s = sum((P[j][i] * Q[i][kk] for i in range(d + 1)),
                        Fraction(0))
                want = Fraction(N) if j == kk else Fraction(0)
                if s != want:
                    return False, f"(PQ)[{j}][{kk}] = {s}, expected {want}"
        return True, (f"P Q = |X| I on a {len(idx)}x{len(idx)} subgrid of "
                      f"{d + 1} eigenspaces, plus structural identities")

    def describe(self) -> str:
        return f"{self.name} (|X| = {self.size}, {self.n_classes} classes)"

    # -- validation gate --------------------------------------------------
    def validate(self, max_exact_classes: int = 64) -> None:
        """Raise unless the scheme is internally consistent.  Memoised."""
        validate_scheme(self, max_exact_classes)

    def portable_identity(self) -> tuple[str, dict]:
        """(kind, JSON-able parameters) letting a verifier REBUILD this scheme.

        Subclasses with a closed form return their defining parameters, so
        an independent verifier derives the eigenmatrices from its own code.
        The base implementation falls back to shipping the tables verbatim,
        which is weaker -- see `ExplicitScheme.portable_identity`.
        """
        return "explicit", {
            "size": self.size,
            "valencies": list(self.valencies()),
            "multiplicities": list(self.multiplicities()),
            "P": [[[int(c.numerator), int(c.denominator)] for c in row]
                  for row in self.P_matrix()],
            "derived": False,
        }


def validate_scheme(scheme: AssociationScheme, max_exact_classes: int = 64
                    ) -> None:
    """Central gate.  Call before anything that can produce a bound.

    TIERED, BECAUSE THE FULL CHECK DOES NOT SCALE.  P Q = |X| I costs
    O(d^3) exact rational operations on integers that themselves grow with
    d: 0.5 s at d = 64, 4 s at d = 128, 44 s at d = 256, and hours by
    d = 1024.  Running it unconditionally would make an asymptotic ladder
    impossible, and silently dropping it would reintroduce exactly the
    failure mode the gate exists to catch.

    So: below `max_exact_classes`, the full product.  Above it, the O(d^2)
    structural identities plus the full product restricted to a subgrid of
    eigenspaces (the low indices, the high indices, and a stride through the
    middle).  A corrupted entry survives only if it misses every sampled row
    and column AND respects the valency, row-sum and Q-row-sum identities.

    That is weaker, and it is the right trade only for schemes with a closed
    form -- whose formulas the test suite validates exactly at small
    parameters, where the full product IS affordable.  For a hand-entered
    `ExplicitScheme` the tables are the only evidence there is, so the full
    check is always run regardless of size.
    """
    if getattr(scheme, "_validated", False):
        return
    ok, message = scheme.check_structure()
    if not ok:
        raise ValueError(f"invalid association scheme {scheme.name}: {message}")

    _, params = scheme.portable_identity()
    derived = bool(params.get("derived", False))
    if scheme.n_classes <= max_exact_classes or not derived:
        ok, message = scheme.check_orthogonality()
    else:
        ok, message = scheme.check_orthogonality_modular()
    if not ok:
        raise ValueError(f"invalid association scheme {scheme.name}: {message}")
    scheme._validated = True


# ---------------------------------------------------------------------------
# Hamming H(n, q):  codes of length n over an alphabet of size q
# ---------------------------------------------------------------------------
class HammingScheme(AssociationScheme):
    r"""Classes = Hamming distance.  Self-dual: Q_{ij} = K_j(i).

        v_i = m_i = C(n,i)(q-1)^i
        P_i(j) = K_i(j) = sum_r (-1)^r (q-1)^{i-r} C(j,r) C(n-j, i-r)
    """

    def __init__(self, n: int, q: int = 2) -> None:
        if n < 1 or q < 2:
            raise ValueError("need n >= 1, q >= 2")
        self.n, self.q = int(n), int(q)
        self.name = f"Hamming H({n},{q})"

    @property
    def n_classes(self) -> int:
        return self.n

    @property
    def size(self) -> int:
        return self.q ** self.n

    @lru_cache(maxsize=None)
    def valencies(self) -> tuple[int, ...]:
        return tuple(math.comb(self.n, i) * (self.q - 1) ** i
                     for i in range(self.n + 1))

    def multiplicities(self) -> tuple[int, ...]:
        return self.valencies()                      # self-dual

    def _fill_P(self) -> list[list[Fraction]]:
        """P[j][i] = K_i(j) by the three-term recurrence in i, O(d^2).

            (i+1) K_{i+1}(x) = [(q-1)(n-i) + i - qx] K_i(x)
                               - (q-1)(n-i+1) K_{i-1}(x)

        Every division is exact because the K_i(x) are integers.
        """
        n, q, d = self.n, self.q, self.n_classes
        rows = []
        for x in range(d + 1):
            row = [1] * (d + 1)
            if d >= 1:
                row[1] = (q - 1) * n - q * x
            for i in range(1, d):
                num = (((q - 1) * (n - i) + i - q * x) * row[i]
                       - (q - 1) * (n - i + 1) * row[i - 1])
                if num % (i + 1):
                    raise ArithmeticError(
                        f"Krawtchouk recurrence produced a non-integer at "
                        f"i={i}, x={x}; the recurrence is wrong")
                row[i + 1] = num // (i + 1)
            rows.append([Fraction(c) for c in row])
        return rows

    def portable_identity(self) -> tuple[str, dict]:
        return "hamming", {"n": self.n, "q": self.q, "derived": True}

    @lru_cache(maxsize=None)
    def eigenvalue(self, i: int, j: int) -> Fraction:
        n, q = self.n, self.q
        return Fraction(sum((-1) ** r * (q - 1) ** (i - r)
                            * math.comb(j, r) * math.comb(n - j, i - r)
                            for r in range(0, i + 1) if i - r <= n - j))


# ---------------------------------------------------------------------------
# Johnson J(n, w):  constant-weight codes
# ---------------------------------------------------------------------------
class JohnsonScheme(AssociationScheme):
    r"""Classes i = w - |A cap B|, so Hamming distance = 2i.

        |X|    = C(n,w)
        v_i    = C(w,i) C(n-w,i)
        m_j    = C(n,j) - C(n,j-1)
        P_i(j) = E_i(j) = sum_r (-1)^r C(j,r) C(w-j, i-r) C(n-w-j, i-r)
                 (Eberlein)

    The dual eigenvalues Q_{ij} are Hahn polynomials; they are RATIONAL, not
    integral, which is exactly why the interface returns Fractions and the
    exact backend carries denominators rather than assuming integers.

    Constant-weight codes are the reason this scheme is here: A(n,d,w) tables
    are far less picked over than binary A(n,d), so systematic exact dual
    search has somewhere to go.
    """

    def __init__(self, n: int, w: int) -> None:
        if not (0 < w <= n):
            raise ValueError("need 0 < w <= n")
        if 2 * w > n:
            raise ValueError(
                f"J({n},{w}) is isomorphic to J({n},{n - w}); pass w <= n/2 so "
                f"the Eberlein binomials stay in range")
        self.n, self.w = int(n), int(w)
        self.name = f"Johnson J({n},{w})"

    @property
    def n_classes(self) -> int:
        return self.w

    @property
    def size(self) -> int:
        return math.comb(self.n, self.w)

    @lru_cache(maxsize=None)
    def valencies(self) -> tuple[int, ...]:
        return tuple(math.comb(self.w, i) * math.comb(self.n - self.w, i)
                     for i in range(self.w + 1))

    @lru_cache(maxsize=None)
    def multiplicities(self) -> tuple[int, ...]:
        return tuple(math.comb(self.n, j) - (math.comb(self.n, j - 1)
                                             if j else 0)
                     for j in range(self.w + 1))

    @lru_cache(maxsize=None)
    def eigenvalue(self, i: int, j: int) -> Fraction:
        n, w = self.n, self.w
        return Fraction(sum((-1) ** r * math.comb(j, r)
                            * math.comb(w - j, i - r)
                            * math.comb(n - w - j, i - r)
                            for r in range(0, i + 1)
                            if 0 <= i - r <= min(w - j, n - w - j)))

    def portable_identity(self) -> tuple[str, dict]:
        return "johnson", {"n": self.n, "w": self.w, "derived": True}

    def hamming_distance(self, i: int) -> int:
        """Hamming distance corresponding to Johnson class i."""
        return 2 * i

    def classes_for_min_distance(self, d: int) -> tuple[int, ...]:
        """Classes i >= 1 whose Hamming distance 2i is at least d."""
        return tuple(i for i in range(1, self.w + 1) if 2 * i >= d)


# ---------------------------------------------------------------------------
# Explicit scheme from user-supplied tables
# ---------------------------------------------------------------------------
class ExplicitScheme(AssociationScheme):
    """A scheme given directly by (|X|, valencies, multiplicities, P).

    The escape hatch for schemes with no closed form here -- q-Johnson,
    bilinear forms, non-symmetric quotients, or a scheme read from a file.
    `check_orthogonality` applies unchanged, so a table typed in by hand is
    still fully validated before any bound is claimed.
    """

    def __init__(self, size: int, valencies, multiplicities, P,
                 name: str = "explicit") -> None:
        self._size = int(size)
        self._v = tuple(int(x) for x in valencies)
        self._m = tuple(int(x) for x in multiplicities)
        self._P = [[Fraction(x) for x in row] for row in P]
        self.name = name
        d = len(self._v) - 1
        if len(self._m) != d + 1 or len(self._P) != d + 1:
            raise ValueError("valencies, multiplicities and P must agree in size")
        if any(len(row) != d + 1 for row in self._P):
            raise ValueError("P must be square")

    @property
    def n_classes(self) -> int:
        return len(self._v) - 1

    @property
    def size(self) -> int:
        return self._size

    def valencies(self) -> tuple[int, ...]:
        return self._v

    def multiplicities(self) -> tuple[int, ...]:
        return self._m

    def eigenvalue(self, i: int, j: int) -> Fraction:
        return self._P[j][i]

    def portable_identity(self) -> tuple[str, dict]:
        """Tables verbatim -- the weaker form of portability.

        A verifier receiving these can confirm P Q = |X| I and every sign
        condition, but it CANNOT confirm that the tables describe the
        combinatorial object you had in mind, because it has no independent
        route to them.  `derived: False` says so, and the verifier surfaces
        it as a warning rather than swallowing it.
        """
        kind, params = super().portable_identity()
        params["name"] = self.name
        return kind, params


SCHEMES = {
    "hamming": HammingScheme,
    "johnson": JohnsonScheme,
}


def _is_small_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    f = 3
    while f * f <= n:
        if n % f == 0:
            return False
        f += 2
    return True
