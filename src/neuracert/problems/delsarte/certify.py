"""Plug-in #2: the Delsarte LP dual over an arbitrary association scheme.

    F(i) = sum_j f_j Q_{ij},   f_0 > 0,  f_j >= 0 (j >= 1),
    F(i) <= 0 for every ACHIEVABLE class i >= 1
    ==>  |C| <= F(0) / f_0 = (sum_j f_j m_j) / f_0

THE SIGN CONDITION LIVES ON THE ACHIEVABLE CLASSES, NOT THE EXCLUDED ONES.
The argument bounds sum_i a_i F(i) above by F(0), which needs F(i) <= 0
exactly where a_i may be nonzero.  Constraining F on the excluded classes
instead constrains it where a_i = 0, proves nothing, and produces
confident-looking "upper bounds" BELOW the true optimum.

SPECTRAL DEGREE, NOT DEGREE
`spectral_degree` is the number of primitive idempotents kept, i.e. the
highest eigenspace index j with f_j allowed nonzero.  In H(n,q) that
coincides with the Krawtchouk polynomial degree, which is why the two are
routinely conflated -- but in a general scheme it is an index into the
eigenspace decomposition and carries no polynomial meaning unless the scheme
is Q-polynomial.  The parameter is named for the invariant thing.

EVERY ENTRY POINT VALIDATES ITS SCHEME
`validate_scheme` (P Q = |X| I, exactly) runs before any bound is produced.
It is memoised per instance, so the cost is one O(d^3) rational pass.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from fractions import Fraction

from neuracert.exact.crt import ModularProblem
from neuracert.exact.schemes import (AssociationScheme, HammingScheme, JohnsonScheme,
                                      validate_scheme)


# ---------------------------------------------------------------------------
# Input normalisation -- every public entry point funnels through this
# ---------------------------------------------------------------------------
def normalise_coeffs(scheme: AssociationScheme, coeffs) -> list[Fraction]:
    """Validate and zero-pad a coefficient vector to length d+1."""
    f = [Fraction(c) for c in coeffs]
    if not f:
        raise ValueError("empty coefficient vector: a dual needs at least f_0")
    d = scheme.n_classes
    if len(f) > d + 1:
        raise ValueError(
            f"{len(f)} coefficients supplied but {scheme.name} has only "
            f"{d + 1} eigenspaces; f_j for j > {d} has no meaning")
    return f + [Fraction(0)] * (d + 1 - len(f))


def normalise_constrained(scheme: AssociationScheme, constrained
                          ) -> tuple[int, ...]:
    """Deduplicate, sort and range-check a constrained class list."""
    cs = tuple(sorted({int(j) for j in constrained}))
    if not cs:
        raise ValueError("empty constrained class set proves nothing")
    d = scheme.n_classes
    bad = [j for j in cs if j < 1 or j > d]
    if bad:
        raise ValueError(f"constrained classes {bad} lie outside 1..{d}")
    return cs


def normalise_min_distance(scheme: AssociationScheme, min_distance):
    """Range-check a minimum distance against the scheme's metric."""
    if min_distance is None:
        return None
    md = int(min_distance)
    if isinstance(scheme, JohnsonScheme):
        hi = 2 * scheme.w
    elif isinstance(scheme, HammingScheme):
        hi = scheme.n
    else:
        raise ValueError(
            f"{type(scheme).__name__} has no distance interpretation; pass "
            f"the achievable classes explicitly and leave min_distance unset")
    if not (1 <= md <= hi):
        raise ValueError(f"min_distance {md} outside 1..{hi} for {scheme.name}")
    return md


def resolve_constrained(scheme: AssociationScheme,
                        min_distance: int | None = None,
                        classes: list[int] | None = None) -> tuple[int, ...]:
    """Classes i >= 1 on which the dual must be non-positive.

    Given explicitly, or derived from a minimum distance.  For Hamming the
    class index IS the distance; for Johnson, class i means Hamming distance
    2i, so the translation is done by the scheme rather than assumed here.
    """
    if classes is not None:
        return normalise_constrained(scheme, classes)
    md = normalise_min_distance(scheme, min_distance)
    if md is None:
        raise ValueError("give either min_distance or an explicit class list")
    if isinstance(scheme, JohnsonScheme):
        cs = scheme.classes_for_min_distance(md)
    else:
        cs = tuple(range(md, scheme.n_classes + 1))
    return normalise_constrained(scheme, cs)


# ---------------------------------------------------------------------------
# Certificates
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DualCertificate:
    """An exactly feasible Delsarte dual and the bound it proves."""

    scheme_name: str
    scheme_size: int
    scheme_kind: str
    scheme_params: dict
    constrained: tuple[int, ...]
    min_distance: int | None
    coeffs: tuple[Fraction, ...]
    bound: Fraction
    integer_bound: int
    repair: dict = field(default_factory=dict)

    @property
    def spectral_degree(self) -> int:
        """Highest eigenspace index with a nonzero coefficient."""
        nz = [j for j, c in enumerate(self.coeffs) if c != 0]
        return max(nz) if nz else 0

    def report(self) -> str:
        return (f"{self.scheme_name}: spectral degree "
                f"{self.spectral_degree} dual, |C| <= {self.bound} = "
                f"{float(self.bound):.6f}, so |C| <= {self.integer_bound}")

    def to_json(self) -> str:
        """Compact, self-contained, verifier-readable."""
        return json.dumps({
            "format": "certkit-delsarte-2",
            "scheme": self.scheme_name,
            "scheme_kind": self.scheme_kind,
            "scheme_params": self.scheme_params,
            "size": self.scheme_size,
            "min_distance": self.min_distance,
            "constrained_classes": list(self.constrained),
            "spectral_degree": self.spectral_degree,
            "coeffs": [[c.numerator, c.denominator] for c in self.coeffs],
            "bound": [self.bound.numerator, self.bound.denominator],
            "integer_bound": self.integer_bound,
            "repair": self.repair,
        }, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Exact feasibility
# ---------------------------------------------------------------------------
def dual_values(scheme: AssociationScheme, coeffs) -> list[Fraction]:
    """F(i) = sum_j f_j Q_{ij} for every class i, exactly."""
    f = normalise_coeffs(scheme, coeffs)
    return [sum((f[j] * scheme.dual_eigenvalue(i, j)
                 for j in range(len(f))), Fraction(0))
            for i in range(scheme.n_classes + 1)]


def check_feasible(scheme: AssociationScheme, coeffs,
                   constrained) -> tuple[bool, list[str]]:
    """Exact feasibility test.  Returns (ok, list of violations)."""
    f = normalise_coeffs(scheme, coeffs)
    cs = normalise_constrained(scheme, constrained)
    bad: list[str] = []
    if f[0] <= 0:
        bad.append(f"f_0 = {f[0]} is not positive")
    for j in range(1, len(f)):
        if f[j] < 0:
            bad.append(f"f_{j} = {f[j]} < 0")
    F = dual_values(scheme, f)
    for i in cs:
        if F[i] > 0:
            bad.append(f"F({i}) = {F[i]} > 0")
    return (not bad), bad


# ---------------------------------------------------------------------------
# Repair
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RepairResult:
    coeffs: list[Fraction]
    method: str                  # "none" | "scale" | "mix"
    parameter: Fraction          # t for scale, lambda for mix
    clamped: int

    def summary(self) -> str:
        if self.method == "none":
            return "already feasible"
        sym = "t" if self.method == "scale" else "lambda"
        return (f"{self.method} repair, {sym} = {float(self.parameter):.12f}, "
                f"{self.clamped} coefficients clamped")

    def as_dict(self) -> dict:
        return {"method": self.method, "clamped": self.clamped,
                "parameter": [self.parameter.numerator,
                              self.parameter.denominator]}


def reference_dual(scheme: AssociationScheme) -> list[Fraction]:
    """The universal repair DIRECTION, available in every scheme for free.

    From Q P = |X| I with P_{j0} = 1 (the relation A_0 = I) comes

        sum_{j=0}^{d} Q_{ij} = |X| delta_{i0}.

    So f* = (0, 1, 1, ..., 1) has

        F*(i) = -1  for every i >= 1,     F*(0) = |X| - 1,     f*_0 = 0.

    Three properties make this the right object to repair with:
      * it lowers F on EVERY constrained class, by exactly 1 per unit;
      * it raises every f_j for j >= 1, so it fixes negative coefficients too;
      * it leaves f_0 alone, so the bound's DENOMINATOR is untouched.

    The last one is why an additive shift beats a convex mix: mixing toward a
    strictly feasible point shrinks f_0 as well, paying twice for the same
    repair.
    """
    return [Fraction(0)] + [Fraction(1)] * scheme.n_classes


def repair_dual(scheme: AssociationScheme, coeffs, constrained,
                verbose: bool = True, prefer: str = "best") -> RepairResult:
    """Map any dual with f_0 > 0 to an EXACTLY feasible one.  Total.

    Two exact strategies; by default both are computed and the one giving
    the BETTER bound is returned.

    SCALE (cheap, partial).  Clamp f_j < 0 to zero for j >= 1, then scale the
    j >= 1 block by the smallest rational t >= 1 meeting every constraint.
    With g(i) = sum_{j>=1} f_j Q_{ij}, the constraint at i is
    f_0 + t g(i) <= 0, solvable exactly when g(i) < 0.  At fine rounding the
    cost is ~1e-14 relative.

    It FAILS when some constrained class has g(i) >= 0 -- not exotic at all:
    it happens whenever clamping removes the coefficients that were
    supplying the negativity at i, i.e. at a degenerate optimum where several
    f_j sit at zero and rounding perturbed the LP's active set.  That is the
    common case at coarse precision, so a partial repair is not enough.

    SHIFT (total).  Add mu * f* with f* = `reference_dual` and

        mu = max( 0, max_i F(i), max_j (-f_j) )

    which is the smallest multiple restoring every condition at once, since
    f* moves each of them at unit rate.  It always exists and is exactly
    rational, so repair never fails.  The cost is additive and known in
    closed form: the bound rises by mu (|X| - 1) / f_0.  That term carries
    |X|, so at coarse rounding a shift-repaired bound can be enormous while
    still perfectly valid -- which is a signal to re-round finer, not to
    distrust the result.  `discover_and_certify` escalates precision
    automatically on exactly this signal.

    `prefer` is "best" (default), "scale", or "shift".
    """
    validate_scheme(scheme)
    f = normalise_coeffs(scheme, coeffs)
    cs = normalise_constrained(scheme, constrained)
    if f[0] <= 0:
        raise ValueError(f"f_0 = {f[0]} must be positive; rescale first")
    if prefer not in ("best", "scale", "shift"):
        raise ValueError(f"prefer must be best/scale/shift, got {prefer!r}")

    ok, _ = check_feasible(scheme, f, cs)
    if ok:
        res = RepairResult(list(f), "none", Fraction(1), 0)
        if verbose:
            print(f"  repair: {res.summary()}")
        return res

    candidates = []
    if prefer in ("best", "scale"):
        sc = _repair_by_scaling(scheme, f, cs)
        if sc is not None:
            candidates.append(sc)
    if prefer in ("best", "shift") or not candidates:
        candidates.append(_repair_by_shift(scheme, f, cs))

    m = scheme.multiplicities()
    def value(r):
        return (sum((r.coeffs[j] * m[j] for j in range(len(r.coeffs))),
                    Fraction(0)) / r.coeffs[0])
    best = min(candidates, key=value)
    if verbose:
        if len(candidates) > 1:
            other = max(candidates, key=value)
            print(f"  repair: {best.summary()}; chose {best.method} over "
                  f"{other.method} ({float(value(best)):.6g} vs "
                  f"{float(value(other)):.6g})")
        else:
            print(f"  repair: {best.summary()}")
    return best


def _repair_by_scaling(scheme, f, cs) -> RepairResult | None:
    g = list(f)
    clamped = sum(1 for j in range(1, len(g)) if g[j] < 0)
    for j in range(1, len(g)):
        if g[j] < 0:
            g[j] = Fraction(0)
    t = Fraction(1)
    for i in cs:
        gi = sum((g[j] * scheme.dual_eigenvalue(i, j)
                  for j in range(1, len(g))), Fraction(0))
        if gi >= 0:
            return None
        need = g[0] / (-gi)
        if need > t:
            t = need
    out = [g[0]] + [t * g[j] for j in range(1, len(g))]
    ok, bad = check_feasible(scheme, out, cs)
    if not ok:  # pragma: no cover - defensive
        raise RuntimeError(f"scaling repair produced an infeasible dual: {bad}")
    return RepairResult(out, "scale", t, clamped)


def _repair_by_shift(scheme, f, cs) -> RepairResult:
    ref = reference_dual(scheme)
    F = dual_values(scheme, f)
    mu = Fraction(0)
    clamped = 0
    for j in range(1, len(f)):
        if f[j] < 0:
            clamped += 1
            mu = max(mu, -f[j])
    for i in cs:
        if F[i] > 0:
            mu = max(mu, F[i])
    out = [f[j] + mu * ref[j] for j in range(len(f))]
    ok, bad = check_feasible(scheme, out, cs)
    if not ok:  # pragma: no cover - defensive
        raise RuntimeError(f"shift repair produced an infeasible dual: {bad}")
    return RepairResult(out, "shift", mu, clamped)


# ---------------------------------------------------------------------------
# Certification
# ---------------------------------------------------------------------------
def certify_dual(scheme: AssociationScheme, coeffs, constrained,
                 min_distance: int | None = None, repair: bool = False,
                 verbose: bool = True) -> DualCertificate:
    """Verify a dual exactly and return the bound it proves.

    `min_distance` is the PROBLEM STATEMENT, and recording it is not
    bookkeeping.  Without it a certificate is only checkable against its own
    `constrained` list, so shrinking that list yields a document that still
    verifies while no longer supporting the headline claim.  With it, an
    independent verifier re-derives the required class set from (scheme,
    min_distance) and rejects any certificate that constrains less.
    """
    validate_scheme(scheme)
    kind, params = scheme.portable_identity()     # fail early, not at the end
    cs = normalise_constrained(scheme, constrained)
    md = normalise_min_distance(scheme, min_distance)
    if md is not None:
        required = set(resolve_constrained(scheme, min_distance=md))
        missing = sorted(required - set(cs))
        if missing:
            raise ValueError(
                f"min_distance {md} requires classes {sorted(required)} to be "
                f"constrained, but {missing} are not; the certificate would "
                f"not support the stated problem")

    f = normalise_coeffs(scheme, coeffs)
    rep = RepairResult(list(f), "none", Fraction(1), 0)
    if repair:
        rep = repair_dual(scheme, f, cs, verbose=verbose)
        f = rep.coeffs

    ok, bad = check_feasible(scheme, f, cs)
    if not ok:
        raise ValueError("dual is infeasible; pass repair=True or fix it:\n  "
                         + "\n  ".join(bad))

    m = scheme.multiplicities()
    F0 = sum((f[j] * m[j] for j in range(len(f))), Fraction(0))
    bound = F0 / f[0]
    return DualCertificate(
        scheme_name=scheme.name, scheme_size=scheme.size, scheme_kind=kind,
        scheme_params=params, constrained=cs, min_distance=md,
        coeffs=tuple(f), bound=bound,
        integer_bound=int(bound.numerator // bound.denominator),
        repair=rep.as_dict())


# ---------------------------------------------------------------------------
# Discovery: the plain LP, which for a finite scheme is the whole story
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LPSolution:
    """A float LP solution in NORMALISED coordinates, plus its exact scale.

    The LP is solved in g_j = m_j f_j rather than f_j directly.  This is not
    cosmetic -- in the raw coordinates the model is unsolvable and the
    rounding is destructive:

      * SOLVABILITY.  The objective coefficients are the m_j and the matrix
        entries are the Q_ij, both of order C(n, n/2).  At n = 64 that is
        1.8e18, past the ~1e15 ceiling above which HiGHS rejects the model
        outright ("Model error").  Every n >= 64 fails.

      * CONDITIONING.  After scaling, the objective is all ones and the
        matrix entry is Q_ij / m_j = P_ji / v_i, which lies in [-1, 1] in
        EVERY association scheme, because an adjacency eigenvalue cannot
        exceed its valency.  So the scaled model is perfectly conditioned at
        any size, in any scheme.

      * ROUNDING.  In raw coordinates f_j ~ 1 / C(n,j) ~ 1e-18, so dyadic
        rounding at 50 bits annihilates every coefficient.  Rounding must
        happen in the g coordinates, where the entries are O(1), and the
        division by the exact integer m_j must come afterwards.

    `to_exact` does exactly that, so the exact coefficients carry denominator
    2^bits * m_j and lose nothing.
    """

    scaled: list[float]
    value: float
    scale: tuple[int, ...]
    spectral_degree: int

    def to_exact(self, bits: int = 50) -> list[Fraction]:
        """Round in the normalised coordinates, then unscale exactly."""
        unit = 1 << int(bits)
        return [Fraction(round(g * unit), unit * int(m))
                for g, m in zip(self.scaled, self.scale)]


def solve_dual_lp(scheme: AssociationScheme, constrained,
                  spectral_degree: int | None = None,
                  zero_classes: list[int] | None = None) -> LPSolution:
    """Float LP for the optimal dual, normalised to f_0 = 1.

    In the scaled variables g_j = m_j f_j:

        minimise  sum_j g_j    s.t.  g_0 = 1,  g_j >= 0 (j >= 1),
                                     sum_j g_j (P_ji / v_i) <= 0, i constrained

    so the objective is all ones and every matrix entry lies in [-1, 1].

    For a finite scheme this is a small dense LP solved to optimality by
    simplex, with no discovery model needed at all.  A neural or structured
    parameterisation only earns its place where the LP stops being small or
    stops being linear: higher-order lifts, q-analogues at large parameters,
    or asymptotic families whose constraint set must be generated rather than
    enumerated.

    `zero_classes` pins chosen g_j to zero, which is how a structural ansatz
    (a prescribed zero pattern) is imposed without leaving the LP.
    """
    try:
        import numpy as np
        from scipy.optimize import linprog
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("solve_dual_lp needs scipy") from exc

    validate_scheme(scheme)
    d = scheme.n_classes if spectral_degree is None else int(spectral_degree)
    if not (1 <= d <= scheme.n_classes):
        raise ValueError(f"spectral_degree {d} outside 1..{scheme.n_classes}")
    cs = normalise_constrained(scheme, constrained)

    zeros = sorted({int(j) for j in (zero_classes or [])})
    bad = [j for j in zeros if j < 1 or j > d]
    if bad:
        raise ValueError(
            f"zero_classes {bad} outside 1..{d}; pinning a coefficient that "
            f"does not exist silently does nothing, so it is an error here")

    m = scheme.multiplicities()
    v = scheme.valencies()
    P = scheme.P_matrix()
    # Q_ij / m_j = P_ji / v_i, in [-1, 1] for every scheme
    A_ub = np.array([[float(Fraction(P[j][i], 1) / Fraction(v[i]))
                      for j in range(d + 1)] for i in cs], dtype=float)
    obj = np.ones(d + 1, dtype=float)
    bounds = [(1.0, 1.0)] + [(0.0, None)] * d
    for j in zeros:
        bounds[j] = (0.0, 0.0)

    res = linprog(obj, A_ub=A_ub, b_ub=np.zeros(len(cs)), bounds=bounds,
                  method="highs")
    if not res.success:
        raise RuntimeError(
            f"LP failed ({res.message}).\n"
            f"    Scheme {scheme.name}, spectral degree {d}, "
            f"{len(cs)} constrained classes.\n"
            f"    The matrix is already normalised into [-1, 1], so this is "
            f"almost certainly the DYNAMIC RANGE OF THE SOLUTION, not the "
            f"model.\n"
            f"    The Delsarte optimum grows like 2^(R n), so the scaled "
            f"variables g_j span that\n"
            f"    range while the solver works to an absolute tolerance near "
            f"1e-7. Double precision\n"
            f"    therefore runs out around 2^53, i.e. n of order 100-200 at "
            f"constant relative\n"
            f"    distance. Beyond that a float LP cannot resolve the "
            f"optimum and will either fail\n"
            f"    (as here) or, worse, return a confident value below the "
            f"true bound.\n"
            f"    Options: reduce n, raise the relative distance, cap "
            f"spectral_degree, or move to an\n"
            f"    exact/high-precision LP. Do not simply retry with looser "
            f"tolerances.")
    return LPSolution(scaled=list(res.x), value=float(res.fun),
                      scale=tuple(int(m[j]) for j in range(d + 1)),
                      spectral_degree=d)


def discover_and_certify(scheme: AssociationScheme, constrained,
                         spectral_degree: int | None = None, bits: int = 50,
                         zero_classes: list[int] | None = None,
                         min_distance: int | None = None,
                         max_bits: int = 400, tolerance: float = 1e-6,
                         verbose: bool = True) -> DualCertificate:
    """Float LP -> dyadic rounding -> exact repair -> exact certificate.

    The rounding happens BEFORE any claim is made and the repair restores
    exact feasibility, so the certificate never depends on the LP solver
    being correct -- only on it being useful.

    PRECISION ESCALATION.  A shift repair costs mu (|X| - 1) / f_0, which
    carries |X| and so can be enormous at coarse rounding even though the
    result is perfectly valid.  Rather than hand back a valid-but-useless
    bound, this doubles `bits` and retries while the certified value exceeds
    the float LP optimum by more than `tolerance` (relative), up to
    `max_bits`.  The escalation only ever affects bound QUALITY -- every
    intermediate certificate was already valid.
    """
    validate_scheme(scheme)
    cs = normalise_constrained(scheme, constrained)
    sol = solve_dual_lp(scheme, cs, spectral_degree, zero_classes)
    val = sol.value

    bits = int(bits)
    best = None
    while True:
        approx = sol.to_exact(bits)
        ok, _ = check_feasible(scheme, approx, cs)
        cert = certify_dual(scheme, approx, cs, min_distance=min_distance,
                            repair=not ok, verbose=False)
        if best is None or cert.bound < best[0].bound:
            best = (cert, bits, ok)
        loss = (float(cert.bound) / val - 1.0) if val else 0.0
        if loss <= tolerance or bits >= max_bits:
            break
        if verbose:
            print(f"  {bits} bits: certified {float(cert.bound):.6g} vs LP "
                  f"{val:.6g} (rel. loss {loss:+.2e}) via "
                  f"{cert.repair['method']} repair -- retrying at "
                  f"{min(2 * bits, max_bits)} bits")
        bits = min(2 * bits, max_bits)

    cert, used_bits, was_ok = best
    if verbose:
        loss = (float(cert.bound) / val - 1.0) if val else 0.0
        print(f"  LP float {val:.10f} -> certified {float(cert.bound):.10f} "
              f"(rel. loss {loss:+.2e}) at {used_bits} bits, rounded dual "
              f"{'was feasible' if was_ok else 'needed ' + cert.repair['method'] + ' repair'}")
    return cert


# ---------------------------------------------------------------------------
# The same bound through the shared CRT engine
# ---------------------------------------------------------------------------
class DelsarteProblem(ModularProblem):
    """F(0) and f_0 as exact integers via the shared multimodular engine.

    Feasibility is NOT part of the modular computation: it is an exact
    rational check done in the constructor, which refuses to build on an
    infeasible dual or an unvalidated scheme.  The engine reconstructs only
    the VALUE.

    For an ordinary finite scheme the direct Fraction path in `certify_dual`
    is faster and this class is redundant -- it exists for the case the
    interface is meant to serve, where the dual's exact coefficients carry
    enough bits that forming F(0) over Z is the bottleneck.
    """

    name = "delsarte_dual"

    def __init__(self, scheme: AssociationScheme, numerators: list[int],
                 denominator: int, constrained) -> None:
        validate_scheme(scheme)
        self.scheme_name = scheme.name
        self.den = int(denominator)
        if self.den <= 0:
            raise ValueError("denominator must be positive")
        self.constrained = normalise_constrained(scheme, constrained)
        coeffs = normalise_coeffs(scheme,
                                  [Fraction(int(v), self.den)
                                   for v in numerators])
        self.num = [int(c * self.den) for c in coeffs]
        ok, bad = check_feasible(scheme, coeffs, self.constrained)
        if not ok:
            raise ValueError("infeasible dual; repair before certifying:\n  "
                             + "\n  ".join(bad))
        self.mult = [int(v) for v in scheme.multiplicities()]

    def functionals(self):
        return ("F0", "f0")

    def a_priori_bounds(self):
        return {"F0": sum(abs(a * b) for a, b in zip(self.num, self.mult)),
                "f0": abs(self.num[0])}

    def denominators(self):
        return {"F0": 1, "f0": 1}

    def residues(self, p: int):
        s = sum(a % p * (b % p) for a, b in zip(self.num, self.mult)) % p
        return {"F0": s, "f0": self.num[0] % p}

    def fingerprint_payload(self):
        return {"scheme": self.scheme_name, "den": self.den,
                "constrained": list(self.constrained), "num": self.num,
                "mult": self.mult}

    def describe(self) -> str:
        return f"Delsarte dual over {self.scheme_name}"

    @staticmethod
    def bound_from(values: dict[str, Fraction]) -> Fraction:
        return values["F0"] / values["f0"]


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------
def binary_code_bound(n: int, d: int, **kw) -> DualCertificate:
    """A(n, d) for binary codes."""
    s = HammingScheme(n, 2)
    return discover_and_certify(s, resolve_constrained(s, min_distance=d),
                                min_distance=d, **kw)


def qary_code_bound(n: int, d: int, q: int, **kw) -> DualCertificate:
    """A_q(n, d)."""
    s = HammingScheme(n, q)
    return discover_and_certify(s, resolve_constrained(s, min_distance=d),
                                min_distance=d, **kw)


def constant_weight_bound(n: int, d: int, w: int, **kw) -> DualCertificate:
    """A(n, d, w) for constant-weight codes, via the Johnson scheme."""
    s = JohnsonScheme(n, w)
    return discover_and_certify(s, resolve_constrained(s, min_distance=d),
                                min_distance=d, **kw)
