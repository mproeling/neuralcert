#!/usr/bin/env python3
"""Independent verifier for certkit Delsarte certificates.

ZERO SHARED CODE.  Imports nothing from certkit and nothing outside the
standard library.  It re-derives the eigenmatrices from the scheme
parameters with its own implementations, re-checks every constraint in exact
rational arithmetic, and recomputes the bound from scratch.

    python verify_delsarte.py cert.json
    python verify_delsarte.py cert.json --quiet     # exit code only

Exit 0 = verified, 1 = rejected, 2 = malformed input.

THE LIMIT OF "INDEPENDENT", AND WHAT IS DONE ABOUT IT
Re-implementing the same Krawtchouk or Eberlein formula from the same
literature convention is not independence: a single mistaken convention
would sit in both the generator and here, and P Q = |X| I would happily hold
for the wrong scheme.  So this file adds checks that do NOT go through those
formulas at all:

  1. CONVENTION-FREE COMBINATORIAL SUMS.  sum_i v_i = |X| via the binomial
     theorem for Hamming and Vandermonde for Johnson; sum_j m_j = |X|;
     v_0 = m_0 = 1.  These fix the valencies without reference to any
     orthogonal polynomial.

  2. STRUCTURAL ROW AND COLUMN IDENTITIES.  P_i(0) = v_i (eigenvalue of A_i
     on the trivial eigenspace) and P_0(j) = 1 (A_0 = I).  A shifted or
     transposed convention breaks these immediately.

  3. A SECOND, INDEPENDENT ROUTE TO Q FOR HAMMING.  H(n,q) is self-dual, so
     Q_{ij} = K_j(i) directly.  Comparing that against m_j P_{ji} / v_i tests
     the symmetry identity C(n,i)(q-1)^i K_j(i) = C(n,j)(q-1)^j K_i(j), which
     a wrong Krawtchouk normalisation does not satisfy.

  4. DUAL ORTHOGONALITY.  sum_i v_i Q_{ij} Q_{ik} = |X| m_j delta_{jk}.

  5. HAND-COMPUTED REFERENCE TABLES.  Literal eigenmatrices for H(3,2) and
     J(5,2), small enough to derive on paper, embedded here and checked
     against the formulas on every run.  A convention error that survives
     1-4 still has to reproduce these exact integers.

For scheme_kind "explicit" the tables are supplied rather than derived, so
checks 1-5 that depend on a closed form do not apply.  That is reported as a
WARNING: such a certificate proves a bound about whatever structure the
tables describe, which the verifier cannot confirm is the object you meant.
"""

from __future__ import annotations

import json
import sys
from fractions import Fraction


# ---------------------------------------------------------------------------
# Independent combinatorics
# ---------------------------------------------------------------------------
def binom(a: int, b: int) -> int:
    """C(a, b), zero outside the valid range.  Own implementation."""
    if b < 0 or a < 0 or b > a:
        return 0
    b = min(b, a - b)
    num = 1
    for i in range(b):
        num = num * (a - i) // (i + 1)
    return num


def krawtchouk(i: int, x: int, n: int, q: int) -> int:
    """K_i(x) for H(n,q), from the defining alternating sum."""
    return sum((-1) ** r * (q - 1) ** (i - r) * binom(x, r)
               * binom(n - x, i - r) for r in range(i + 1))


def eberlein(i: int, x: int, n: int, w: int) -> int:
    """E_i(x) for J(n,w), from the defining alternating sum."""
    return sum((-1) ** r * binom(x, r) * binom(w - x, i - r)
               * binom(n - w - x, i - r) for r in range(i + 1))


# ---------------------------------------------------------------------------
# Hand-computed reference tables (check 5)
# ---------------------------------------------------------------------------
# Small enough to derive on paper; embedded as literals so a shared
# convention error in the formulas above cannot pass silently.
REFERENCE_TABLES = {
    ("hamming", 3, 2): {
        "size": 8,
        "v": [1, 3, 3, 1],
        "m": [1, 3, 3, 1],
        "P": [[1, 3, 3, 1], [1, 1, -1, -1], [1, -1, -1, 1], [1, -3, 3, -1]],
    },
    ("johnson", 5, 2): {
        "size": 10,
        "v": [1, 6, 3],
        "m": [1, 4, 5],
        "P": [[1, 6, 3], [1, 1, -2], [1, -2, 1]],
    },
}


def check_reference_tables(log) -> bool:
    """Reproduce the hand-computed tables from the formulas."""
    for key, ref in REFERENCE_TABLES.items():
        kind = key[0]
        if kind == "hamming":
            n, q = key[1], key[2]
            v = [binom(n, i) * (q - 1) ** i for i in range(n + 1)]
            m = list(v)
            P = [[krawtchouk(i, j, n, q) for i in range(n + 1)]
                 for j in range(n + 1)]
            size = q ** n
        else:
            n, w = key[1], key[2]
            v = [binom(w, i) * binom(n - w, i) for i in range(w + 1)]
            m = [binom(n, j) - (binom(n, j - 1) if j else 0)
                 for j in range(w + 1)]
            P = [[eberlein(i, j, n, w) for i in range(w + 1)]
                 for j in range(w + 1)]
            size = binom(n, w)
        if (size, v, m, P) != (ref["size"], ref["v"], ref["m"], ref["P"]):
            log(f"REJECT: this verifier's own formulas disagree with the "
                f"hand-computed reference table for {key}; the build is "
                f"broken, not the certificate")
            return False
    log(f"  self-check: formulas reproduce {len(REFERENCE_TABLES)} "
        f"hand-computed reference eigenmatrices")
    return True


# ---------------------------------------------------------------------------
# Scheme reconstruction
# ---------------------------------------------------------------------------
def scheme_tables(kind: str, params: dict):
    """(size, valencies, multiplicities, P, derived)."""
    if kind == "hamming":
        n, q = int(params["n"]), int(params["q"])
        v = [binom(n, i) * (q - 1) ** i for i in range(n + 1)]
        return (q ** n, v, list(v),
                [[Fraction(krawtchouk(i, j, n, q)) for i in range(n + 1)]
                 for j in range(n + 1)], True)
    if kind == "johnson":
        n, w = int(params["n"]), int(params["w"])
        v = [binom(w, i) * binom(n - w, i) for i in range(w + 1)]
        m = [binom(n, j) - (binom(n, j - 1) if j else 0) for j in range(w + 1)]
        return (binom(n, w), v, m,
                [[Fraction(eberlein(i, j, n, w)) for i in range(w + 1)]
                 for j in range(w + 1)], True)
    if kind == "explicit":
        return (int(params["size"]), [int(x) for x in params["valencies"]],
                [int(x) for x in params["multiplicities"]],
                [[Fraction(a, b) for a, b in row] for row in params["P"]],
                False)
    raise ValueError(f"unknown scheme kind {kind!r}")


def structural_checks(kind, params, size, v, m, P, log) -> bool:
    """Checks 1 and 2: convention-free sums and structural identities."""
    d = len(v) - 1
    if v[0] != 1 or m[0] != 1:
        log(f"REJECT: v_0 = {v[0]}, m_0 = {m[0]}; both must be 1")
        return False
    if sum(v) != size:
        log(f"REJECT: sum of valencies {sum(v)} != |X| = {size}")
        return False
    if sum(m) != size:
        log(f"REJECT: sum of multiplicities {sum(m)} != |X| = {size}")
        return False
    for i in range(d + 1):
        if P[0][i] != v[i]:
            log(f"REJECT: P_{i}(0) = {P[0][i]} != v_{i} = {v[i]}")
            return False
    for j in range(d + 1):
        if P[j][0] != 1:
            log(f"REJECT: P_0({j}) = {P[j][0]} != 1 (A_0 must be the identity)")
            return False

    # Check 1, in its convention-free form
    if kind == "hamming":
        n, q = int(params["n"]), int(params["q"])
        if sum(v) != q ** n:
            log("REJECT: valencies do not sum to q^n")
            return False
        note = "sum_i C(n,i)(q-1)^i = q^n"
    elif kind == "johnson":
        n, w = int(params["n"]), int(params["w"])
        if sum(v) != binom(n, w):
            log("REJECT: valencies do not sum to C(n,w)")
            return False
        note = "sum_i C(w,i)C(n-w,i) = C(n,w) (Vandermonde)"
    else:
        note = "supplied tables"
    log(f"  structural: v_0 = m_0 = 1, P_i(0) = v_i, P_0(j) = 1, {note}")
    return True


def dual_eigenmatrix(kind, params, size, v, m, P, log):
    """Q_{ij} = m_j P_{ji}/v_i, cross-checked three ways (checks 3 and 4)."""
    d = len(v) - 1
    Q = [[Fraction(m[j]) * P[j][i] / Fraction(v[i]) for j in range(d + 1)]
         for i in range(d + 1)]

    for j in range(d + 1):
        for k in range(d + 1):
            s = sum((P[j][i] * Q[i][k] for i in range(d + 1)), Fraction(0))
            want = Fraction(size) if j == k else Fraction(0)
            if s != want:
                raise ValueError(f"(PQ)[{j}][{k}] = {s}, expected {want}")

    # Check 3: for Hamming, self-duality gives Q independently
    if kind == "hamming":
        n, q = int(params["n"]), int(params["q"])
        for i in range(d + 1):
            for j in range(d + 1):
                if Q[i][j] != Fraction(krawtchouk(j, i, n, q)):
                    raise ValueError(
                        f"self-duality fails: Q[{i}][{j}] = {Q[i][j]} but "
                        f"K_{j}({i}) = {krawtchouk(j, i, n, q)}")
        log("  independence: Q re-derived via H(n,q) self-duality "
            "(Q_ij = K_j(i)) agrees with m_j P_ji / v_i")

    # Check 4: dual orthogonality
    for j in range(d + 1):
        for k in range(d + 1):
            s = sum((Fraction(v[i]) * Q[i][j] * Q[i][k]
                     for i in range(d + 1)), Fraction(0))
            want = Fraction(size * m[j]) if j == k else Fraction(0)
            if s != want:
                raise ValueError(
                    f"dual orthogonality fails at (j,k) = ({j},{k}): "
                    f"{s} != {want}")
    log(f"  eigenmatrices: P Q = |X| I and sum_i v_i Q_ij Q_ik = |X| m_j "
        f"delta_jk, both exact")
    return Q


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
SUPPORTED = ("certkit-delsarte-1", "certkit-delsarte-2")


def verify(doc: dict, log=print) -> bool:
    if doc.get("format") not in SUPPORTED:
        log(f"REJECT: unknown format {doc.get('format')!r}")
        return False
    if not check_reference_tables(log):
        return False

    kind = doc["scheme_kind"]
    params = doc["scheme_params"]
    size, v, m, P, derived = scheme_tables(kind, params)
    if size != doc["size"]:
        log(f"REJECT: |X| = {size} but certificate claims {doc['size']}")
        return False
    if not derived:
        log(f"  WARNING: scheme '{params.get('name', kind)}' ships its tables "
            f"verbatim; this verifier can confirm they form a consistent "
            f"scheme but has no independent route to them, so it cannot "
            f"confirm they describe the intended combinatorial object")
    if not structural_checks(kind, params, size, v, m, P, log):
        return False
    Q = dual_eigenmatrix(kind, params, size, v, m, P, log)
    d = len(v) - 1

    f = [Fraction(a, b) for a, b in doc["coeffs"]]
    if not f:
        log("REJECT: empty coefficient vector")
        return False
    if len(f) > d + 1:
        log(f"REJECT: {len(f)} coefficients for a scheme with {d + 1} "
            f"eigenspaces")
        return False
    f = f + [Fraction(0)] * (d + 1 - len(f))

    if f[0] <= 0:
        log(f"REJECT: f_0 = {f[0]} is not positive")
        return False
    for j in range(1, d + 1):
        if f[j] < 0:
            log(f"REJECT: f_{j} = {f[j]} < 0")
            return False
    log(f"  coefficient signs: f_0 > 0 and f_j >= 0 for all {d} others")

    constrained = sorted({int(i) for i in doc["constrained_classes"]})
    if not constrained:
        log("REJECT: empty constrained class set proves nothing")
        return False
    if any(i < 1 or i > d for i in constrained):
        log(f"REJECT: constrained classes must lie in 1..{d}")
        return False

    # The certificate's own class list is not trustworthy on its own: a
    # narrowed list still verifies internally while no longer supporting the
    # stated problem.  Re-derive the required set from (scheme, distance).
    min_d = doc.get("min_distance")
    if min_d is None:
        log("  WARNING: no min_distance recorded, so the constrained set "
            "cannot be checked against a problem statement; this certificate "
            "proves a bound only for codes meeting exactly the listed classes")
    else:
        min_d = int(min_d)
        if kind == "hamming":
            if not (1 <= min_d <= d):
                log(f"REJECT: min_distance {min_d} outside 1..{d}")
                return False
            required = set(range(min_d, d + 1))
        elif kind == "johnson":
            if not (1 <= min_d <= 2 * d):
                log(f"REJECT: min_distance {min_d} outside 1..{2 * d}")
                return False
            required = {i for i in range(1, d + 1) if 2 * i >= min_d}
        else:
            log(f"REJECT: cannot derive the required class set for {kind!r}")
            return False
        missing = sorted(required - set(constrained))
        if missing:
            log(f"REJECT: minimum distance {min_d} requires F(i) <= 0 on "
                f"classes {sorted(required)}, but {missing} are unconstrained")
            return False
        log(f"  problem statement: minimum distance {min_d} requires "
            f"{len(required)} constrained classes, all present")

    worst = None
    for i in constrained:
        Fi = sum((f[j] * Q[i][j] for j in range(d + 1)), Fraction(0))
        if Fi > 0:
            log(f"REJECT: F({i}) = {Fi} > 0")
            return False
        if worst is None or Fi > worst[1]:
            worst = (i, Fi)
    log(f"  distance constraints: F(i) <= 0 on all {len(constrained)} "
        f"constrained classes (tightest F({worst[0]}) = "
        f"{float(worst[1]):.6e})")

    F0 = sum((f[j] * Fraction(m[j]) for j in range(d + 1)), Fraction(0))
    bound = F0 / f[0]
    claimed = Fraction(doc["bound"][0], doc["bound"][1])
    if bound != claimed:
        log(f"REJECT: recomputed bound {bound} != claimed {claimed}")
        return False
    if bound.numerator // bound.denominator != doc["integer_bound"]:
        log(f"REJECT: floor is {bound.numerator // bound.denominator}, "
            f"certificate claims {doc['integer_bound']}")
        return False

    log(f"  bound: F(0)/f_0 = {bound} = {float(bound):.10f}")
    log(f"VERIFIED: |C| <= {doc['integer_bound']}")
    return True


def main(argv) -> int:
    args = [a for a in argv[1:] if not a.startswith("-")]
    quiet = "--quiet" in argv
    if len(args) != 1:
        print(__doc__)
        return 2
    try:
        with open(args[0]) as fh:
            doc = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"cannot read certificate: {exc}")
        return 2
    log = (lambda *a, **k: None) if quiet else print
    try:
        ok = verify(doc, log)
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        log(f"REJECT: malformed or inconsistent certificate: {exc}")
        return 1
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))


def cli_main() -> int:
    """Console-script adapter that preserves the verifier's standalone API."""
    return main(sys.argv)
