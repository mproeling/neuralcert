#!/usr/bin/env python3
"""
Discovery playground for the radial sign-uncertainty problem A_s(d)
in a *Gaussian-mixture* Fourier-eigenfunction family.

--------------------------------------------------------------------------
THE PROBLEM (Cohn-Goncalves, Problem 1.1 / 1.3)

    s in {+1,-1}.  Find the smallest rho >= 0 for which there is a radial
    f : R^d -> R, f not identically 0, with

        (1) fhat = s f          (Fourier eigenfunction, unitary convention)
        (2) f(0) = 0
        (3) f(x) >= 0 for |x| >= rho.

    s=+1 is Bourgain-Clozel-Kahane; s=-1 is the sphere-packing / modular
    bootstrap variant.  A_+(12) = A_-(8) = sqrt(2) are the only sharp values
    known besides A_-(1)=1, A_-(24)=2.

--------------------------------------------------------------------------
WHY THIS FAMILY, AND WHY IT IS NOT KILLED BY COHN-DONG-GONCALVES

Everybody's ansatz is  f(x) = p(2 pi |x|^2) e^{-pi|x|^2}  with p a
polynomial (equivalently: a finite Laguerre expansion).  Cohn, Dong and
Goncalves (arXiv:2210.01684) proved that for deg(p) = o(d) this ansatz
cannot beat rho ~ sqrt(d / 2 pi) as d -> oo: the whole class is
asymptotically stuck at what degree three already gives.

We instead use mixtures of Gaussians of *different widths*.  With the
unitary convention F[e^{-pi a |x|^2}](xi) = a^{-d/2} e^{-pi |xi|^2 / a},
put

    phi_{a,s}(r) = e^{-pi a r^2} + s a^{-d/2} e^{-pi r^2 / a}.

Then F[phi_{a,s}] = s phi_{a,s} EXACTLY, for every a > 0 and either sign,
before any optimisation.  So

    f = sum_j c_j phi_{a_j, s}

is an exact s-eigenfunction with 2k free real parameters: k coefficients
c_j AND k widths a_j.  Three structural facts:

  * It is NOT of the form (bounded-degree polynomial) x (fixed Gaussian).
    A single e^{-pi a r^2} with a != 1 is an infinite Laguerre series.
    The CDG hypothesis simply does not apply.  This is the escape hatch.

  * The family is asymptotically complete.  In the variable x = e^{-pi r^2}
    the mixtures span {x^a : a > 0}, an algebra separating points of (0,1]
    and vanishing nowhere, hence dense in C_0((0,1]) by Stone-Weierstrass;
    projecting with (1 + s F)/2 gives density in the s-eigenspace.  So
    sup over k of what this family achieves is A_s(d) itself.  Unlike the
    bounded-degree polynomial class, there is no a priori ceiling.

  * For rational widths and even d it stays exactly certifiable: with
    L = lcm of denominators/numerators and x = e^{-pi r^2 / L}, f becomes
    a genuine rational polynomial in x on [0, x0], so nonnegativity on a
    ray is a univariate rational SOS / interval problem.

The honest risk (the experiment to run FIRST, see --mode asymptotic):
exponential sums {e^{-a u}} form an extended Chebyshev system on (0,oo),
so Krein's generalised Gauss quadrature might reproduce the CDG
obstruction for k = o(d) widths, with the width count playing the role of
the degree.  If rho * sqrt(2 pi / d) refuses to drop below 1 as d grows
with k fixed, that is the Chebyshev-system obstruction showing itself and
the interesting regime is k growing with d.

--------------------------------------------------------------------------
ELEMENTARY FACTS BAKED INTO THE CODE (all are traps in the naive version)

  * phi_{1/a,s} = s a^{d/2} phi_{a,s}.  Widths a and 1/a give the SAME
    basis element up to a scalar => restrict to a >= 1.
  * phi_{1,-1} = 0 identically.  a = 1 is legal only for s = +1.
  * phi_{a,s}(0) = 1 + s a^{-d/2} > 0 for every admissible a, so the
    constraint f(0) = 0 forces the coefficient vector to change sign; f
    therefore has at least one root, as it must.
  * The slowest-decaying term is s c_{jmax} a_max^{-d/2} e^{-pi r^2/a_max}
    with a_max = max_j a_j, so eventual nonnegativity needs
    s * c_{jmax} > 0 (or a degenerate solution with c_{jmax} = 0).
  * Descartes for exponential sums: writing f = sum_i gamma_i x^{beta_i}
    with x = e^{-pi r^2}, the number of positive roots (with multiplicity)
    is at most the number of sign changes in (gamma_i) ordered by beta_i.
    With k widths there are 2k terms, hence at most 2k-1 roots.  Since
    extremisers must have infinitely many double roots (Goncalves-Oliveira
    e Silva-Steinerberger Thm 4, Cohn-Goncalves Thm 1.4), no finite k is
    ever extremal -- but each k gives a legitimate upper bound, and the
    root count tells us how close to "saturated" a solution is.
  * There is NO dilation freedom left.  Rescaling breaks fhat = s f, so
    rho is a genuine number, not a normalisation choice.

--------------------------------------------------------------------------
WHAT THIS SCRIPT IS AND IS NOT

It is DISCOVERY code.  Nonnegativity is imposed and checked on finite
grids, plus an a posteriori tail-dominance check for the unbounded part
of the ray.  It is NOT a certificate.  No neural anything, no interval
arithmetic yet -- the point is to find out whether this family is worth
certifying at all.

References:
  [BCK]  Bourgain, Clozel, Kahane, arXiv:0811.4360
  [GOS]  Goncalves, Oliveira e Silva, Steinerberger, arXiv:1602.03366
  [CG]   Cohn, Goncalves, Invent. Math. 217 (2019), arXiv:1712.04438
  [CDG]  Cohn, Dong, Goncalves, arXiv:2210.01684
  [E]    Edwin, arXiv:2505.15994

Dependencies: numpy, scipy; matplotlib only for --plot.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, linprog, minimize, minimize_scalar
from scipy.special import gammaln, jv


# ---------------------------------------------------------------------------
# Published benchmarks
# ---------------------------------------------------------------------------

# [CG] Table 4.1, first column: rigorous upper bounds for A_+(d).
# d = 12 is exact (sqrt 2).  d <= 2 are the weakest (k = 5 only).
PUBLISHED_APLUS = {
    1: 0.572990, 2: 0.756207, 3: 0.887864, 4: 0.965953,
    5: 1.036454, 6: 1.101116, 7: 1.161109, 8: 1.217275,
    9: 1.270241, 10: 1.320483, 11: 1.368375, 12: math.sqrt(2.0),
    13: 1.458239, 14: 1.500647, 15: 1.541603, 16: 1.581246,
    17: 1.619692, 18: 1.657044, 19: 1.693390, 20: 1.728806,
    21: 1.763360, 22: 1.797112, 23: 1.830115, 24: 1.862417,
    25: 1.894060, 26: 1.925084, 27: 1.955522, 28: 1.985407,
    29: 2.014769, 30: 2.043633, 31: 2.072024, 32: 2.099965,
}

# [CG] Table 4.1, second column, re-indexed: upper bounds for A_-(d).
# d = 1, 8, 24 are exact (1, sqrt 2, 2); d = 2 is 2/sqrt(3) to 1000 digits.
PUBLISHED_AMINUS = {
    1: 1.0, 2: 1.074570, 3: 1.141962, 4: 1.203808,
    5: 1.261244, 6: 1.315083, 7: 1.365923, 8: math.sqrt(2.0),
    9: 1.460307, 10: 1.504478, 11: 1.546952, 12: 1.587911,
    13: 1.627509, 14: 1.665874, 15: 1.703115, 16: 1.739328,
    17: 1.774593, 18: 1.808982, 19: 1.842559, 20: 1.875378,
    21: 1.907490, 22: 1.938938, 23: 1.969763, 24: 2.0,
    25: 2.029684, 26: 2.058842, 27: 2.087503, 28: 2.115691,
}

EXACT_VALUES = {(1, 12), (-1, 1), (-1, 8), (-1, 24)}


def published_upper(d: int, s: int) -> float | None:
    table = PUBLISHED_APLUS if s == 1 else PUBLISHED_AMINUS
    return table.get(d)


def lb_volume(d: int) -> float:
    """[CG] (3.2): A_s(d) >= (1 / (2 vol B_1^d))^{1/d}.  Valid for both signs."""
    log_val = gammaln(d / 2 + 1) - math.log(2.0)
    return math.exp(log_val / d) / math.sqrt(math.pi)


def _first_bessel_zero(nu: float) -> float:
    """Smallest positive zero of J_nu, nu >= 0, by McMahon bracket + brentq."""
    guess = nu + 1.8557571 * nu ** (1.0 / 3.0) + 2.0
    lo, hi = max(nu, 1e-6), guess
    while jv(nu, hi) > 0:
        hi *= 1.5
        if hi > 1e6:
            raise RuntimeError("no Bessel zero bracketed")
    while jv(nu, lo) <= 0 and lo > 1e-9:
        lo *= 0.5
    return brentq(lambda t: jv(nu, t), lo, hi, xtol=1e-13, rtol=1e-14)


def lb_gos(d: int) -> float:
    """
    [GOS] Theorem 7 (s = +1 only): A_+(d) >= (1/sqrt pi) (Gamma(d/2+1)/(1+lam_d))^{1/d},
    lam_d = -2^{d/2} Gamma(d/2+1) J_{d/2}(j_{d/2+1}) / j_{d/2+1}^{d/2}.
    """
    nu = d / 2.0
    j = _first_bessel_zero(nu + 1.0)
    log_lam = (d / 2.0) * math.log(2.0) + gammaln(d / 2 + 1) - (d / 2.0) * math.log(j)
    lam = -math.exp(log_lam) * jv(nu, j)
    return math.exp((gammaln(d / 2 + 1) - math.log1p(lam)) / d) / math.sqrt(math.pi)


def lb_edwin(d: int) -> float:
    """[E] Theorem 1.4 / (1.7), both signs, sharper than lb_volume for d >= 5."""
    return math.sqrt(math.e / (2 * math.pi)) * math.exp(
        (gammaln(d / 2 + 1) - math.log(4.0)) / d
    )


def best_lower_bound(d: int, s: int) -> tuple[float, str]:
    cands = [(lb_volume(d), "volume/[CG](3.2)")]
    if d >= 5:
        cands.append((lb_edwin(d), "Edwin [E] Thm 1.4"))
    if s == 1 and d >= 2:
        try:
            cands.append((lb_gos(d), "GOS [GOS] Thm 7"))
        except Exception:
            pass
    return max(cands, key=lambda t: t[0])


def bck_upper(d: int) -> float:
    """[BCK] (3.10): B_d <= (d+2)/2pi, i.e. A_+(d) <= sqrt((d+2)/2pi).

    This is exactly what the three-Gaussian member of our family (widths
    1, a, 1/a with a -> 1) gives, so it is the natural 'k = 2' baseline.
    """
    return math.sqrt((d + 2) / (2 * math.pi))


def cdg_asymptote(d: int) -> float:
    """[CDG] Theorem 1.2: sqrt(d / 2pi), the wall for sublinear-degree polys."""
    return math.sqrt(d / (2 * math.pi))


# ---------------------------------------------------------------------------
# The eigenfunction family
# ---------------------------------------------------------------------------

@dataclass
class Family:
    """f(u) = sum_j c_j [ e^{-a_j u} + s a_j^{-d/2} e^{-u/a_j} ],  u = pi r^2."""

    d: int
    s: int
    scales: np.ndarray

    def __post_init__(self):
        a = np.asarray(self.scales, dtype=float).ravel()
        if np.any(a <= 0):
            raise ValueError("widths must be positive")
        # phi_{1/a} is proportional to phi_a: fold everything to a >= 1.
        a = np.where(a < 1.0, 1.0 / a, a)
        a = np.sort(np.unique(np.round(a, 12)))
        if self.s == -1 and abs(a[0] - 1.0) < 1e-9:
            raise ValueError("width a = 1 gives the zero function when s = -1")
        if len(a) < 2:
            raise ValueError("need at least two distinct widths")
        object.__setattr__(self, "scales", a)

    # -- basic quantities --------------------------------------------------
    @property
    def k(self) -> int:
        return len(self.scales)

    @property
    def log_w(self) -> np.ndarray:
        """log of a_j^{-d/2}; the tail term is s * exp(log_w) * e^{-u/a}."""
        return -0.5 * self.d * np.log(self.scales)

    @property
    def nu(self) -> float:
        """Slowest decay rate present, 1/a_max."""
        return 1.0 / self.scales[-1]

    # -- EVERYTHING below evaluates the RESCALED function ------------------
    #    g(u) = e^{nu u} f(u),  nu = 1/a_max.
    # Every exponent in g is >= 0, so g is bounded by sum |gamma_i| on the
    # whole ray: no underflow, ever.  Evaluating f itself is hopeless for
    # large d, where the interesting region sits at u ~ d/2 and every term
    # of f is below 1e-300 long before the tail comparison is settled.
    # sign(g) = sign(f), so all positivity questions are unaffected.

    def basis(self, u) -> np.ndarray:
        """(n, k) matrix of e^{nu u} phi_{a_j,s}(u)."""
        u = np.atleast_1d(np.asarray(u, dtype=float))
        nu = self.nu
        head = np.exp(-np.outer(u, self.scales - nu))
        tail = self.s * np.exp(self.log_w[None, :]
                               - np.outer(u, 1.0 / self.scales - nu))
        return head + tail

    def dbasis(self, u) -> np.ndarray:
        """d/du of the rescaled basis (not of phi itself)."""
        u = np.atleast_1d(np.asarray(u, dtype=float))
        nu = self.nu
        hr = self.scales - nu
        tr = 1.0 / self.scales - nu
        head = -hr[None, :] * np.exp(-np.outer(u, hr))
        tail = -(self.s * tr)[None, :] * np.exp(
            self.log_w[None, :] - np.outer(u, tr))
        return head + tail

    def eval(self, u, coeffs) -> np.ndarray:
        return self.basis(u) @ np.asarray(coeffs, dtype=float)

    def deval(self, u, coeffs) -> np.ndarray:
        return self.dbasis(u) @ np.asarray(coeffs, dtype=float)

    def eval_r(self, r, coeffs) -> np.ndarray:
        return self.eval(math.pi * np.asarray(r, dtype=float) ** 2, coeffs)

    # -- structure ---------------------------------------------------------
    def terms(self, coeffs) -> tuple[np.ndarray, np.ndarray]:
        """All 2k (exponent, coefficient) pairs of f, sorted by exponent."""
        c = np.asarray(coeffs, dtype=float)
        beta = np.concatenate([self.scales, 1.0 / self.scales])
        gamma = np.concatenate([c, self.s * np.exp(self.log_w) * c])
        order = np.argsort(beta)
        return beta[order], gamma[order]

    def descartes_bound(self, coeffs, tol_rel=1e-11) -> int:
        """Max number of positive roots (with multiplicity) of f."""
        _, gamma = self.terms(coeffs)
        keep = np.abs(gamma) > tol_rel * np.max(np.abs(gamma))
        sgn = np.sign(gamma[keep])
        return int(np.sum(sgn[1:] != sgn[:-1]))

    def term_magnitude(self, u, coeffs) -> np.ndarray:
        """
        sum_i |gamma_i| e^{-beta_i u}: the size of the terms before
        cancellation.  Floating-point error in evaluating f at u is a few
        machine epsilons times this, so it -- not the global max of |f| -- is
        the right yardstick for "is this dip real or is it noise?".
        """
        beta, gamma = self.terms(coeffs)
        u = np.atleast_1d(np.asarray(u, dtype=float))
        return np.exp(-np.outer(u, beta - self.nu)) @ np.abs(gamma)

    def mass_functional(self, u0: float) -> np.ndarray:
        """m . c = e^{nu u0} int_{u0}^oo f.  Strictly positive on the cone."""
        a, nu = self.scales, self.nu
        head = np.exp(-(a - nu) * u0) / a
        tail = self.s * np.exp(self.log_w - (1.0 / a - nu) * u0) * a
        return head + tail   # = e^{nu u0} int_{u0}^oo f, same sign, no underflow

    def tail_dominance(self, coeffs, safety=1e3, rel_floor=1e-13):
        """
        Smallest u beyond which the slowest surviving term provably dominates
        every other term, so f > 0 on [u, oo) with no grid at all.

        Returns (u_dom, lead_rel, verdict) with verdict in
          "ok"        leading coefficient positive and not marginal,
          "negative"  f is eventually NEGATIVE -- reject outright,
          "marginal"  the leading coefficient is below rel_floor times the
                      largest one, so double precision cannot tell its sign.
                      Exactly the situation an exact certificate would have to
                      settle; in discovery we refuse to trust it.
        """
        beta, gamma = self.terms(coeffs)
        gmax = float(np.max(np.abs(gamma)))
        if gmax == 0.0:
            return math.inf, 0.0, "negative"
        idx = np.where(np.abs(gamma) > rel_floor * gmax)[0]
        if len(idx) == 0:
            return math.inf, 0.0, "marginal"
        j = int(idx[0])
        lead_beta, lead_gamma = beta[j], gamma[j]
        lead_rel = abs(lead_gamma) / gmax
        if lead_gamma <= 0:
            return math.inf, lead_rel, "negative"
        # terms slower than the leading one exist but are below the floor:
        # their sign is not resolvable in double precision.
        verdict = "marginal" if j > 0 else "ok"
        us = [0.0]
        for b, g in zip(beta, np.abs(gamma)):
            if g <= 0 or b <= lead_beta:
                continue
            us.append(math.log(safety * g / lead_gamma) / (b - lead_beta))
        return max(us), lead_rel, verdict

    def tail_dominance_u(self, coeffs, safety=1e3) -> float:
        return self.tail_dominance(coeffs, safety=safety)[0]


def make_scales(k: int, s: int, spread: float = 1.6) -> np.ndarray:
    """Geometric default widths: 1, r, r^2, ... (starting at r for s = -1)."""
    if s == 1:
        return spread ** np.arange(k, dtype=float)
    return spread ** np.arange(1, k + 1, dtype=float)



# ---------------------------------------------------------------------------
# Grid-free positivity: branch and bound on monotone terms
# ---------------------------------------------------------------------------

def terms_xp(fam: Family, c) -> tuple[np.ndarray, np.ndarray]:
    """Shifted exponents and coefficients of g = e^{nu u} f, in longdouble."""
    a = np.asarray(fam.scales, dtype=XP)
    nu = XP(1.0) / a[-1]
    c = np.asarray(c, dtype=XP)
    beta = np.concatenate([a, XP(1.0) / a]) - nu
    gamma = np.concatenate([c, XP(fam.s) * a ** (-XP(fam.d) / 2) * c])
    o = np.argsort(beta)
    return beta[o], gamma[o]


def certified_min_xp(fam: Family, c, u_lo: float, u_hi: float,
                     tol_rel: float = 1e-14, max_boxes: int = 400000,
                     max_rounds: int = 300, width_floor: float = 1e-12):
    """
    Lower-bound g on [u_lo, u_hi] by adaptive subdivision -- no sampling grid.

    A fixed verification grid CANNOT be trusted here, and this is not a
    tolerance question.  Near a double root that has split by a hair, the two
    simple roots sit within ~1e-6 of each other with a lobe between them of
    depth ~1e-7; a grid node landing on the (near-zero) tangency reports
    "clean" while the lobe next door is real.  Cohn-Goncalves flag exactly
    this splitting as what breaks their numerics, and refining the grid only
    moves the goalposts -- 6000 nodes said -2.5e-18, 8000 said -3.2e-7,
    18000 said -5.7e-7 on the same function.

    Here every term of g is monotone in u, so on a box [alpha, beta] the
    centred form gives a genuine lower bound

        g >= g(mid) - (width/2) * sum_i |gamma_i| beta_i e^{-beta_i alpha},

    which converges quadratically in the box width.  Boxes that cannot be
    cleared are split; boxes narrower than width_floor are recorded as
    unresolved (that is what a true tangency looks like) rather than reported
    as violations.  A negative value of g(mid) is a witness: a point the LP
    must be given as a cut.
    """
    beta, gamma = terms_xp(fam, c)
    ag = np.abs(gamma)
    agb = ag * beta
    lo = np.array([u_lo], dtype=XP)
    hi = np.array([u_hi], dtype=XP)
    best_rel, best_u, witness = math.inf, float(u_lo), None
    unresolved = 0

    for _ in range(max_rounds):
        mid = (lo + hi) / 2
        w = hi - lo
        E_lo = np.exp(-np.outer(lo, beta))
        scale = np.maximum(E_lo @ ag, XP(1e-4930))
        fmid = np.exp(-np.outer(mid, beta)) @ gamma
        rel_mid = fmid / scale
        j = int(np.argmin(rel_mid))
        if float(rel_mid[j]) < best_rel:
            best_rel, best_u = float(rel_mid[j]), float(mid[j])
        if float(rel_mid[j]) < -tol_rel and witness is None:
            witness = float(mid[j])
        lower = fmid - (w / 2) * (E_lo @ agb)
        bad = (lower / scale) < -tol_rel
        thin = w <= XP(width_floor) * np.maximum(hi, XP(1.0))
        unresolved += int(np.sum(bad & thin))
        bad &= ~thin
        n_bad = int(np.sum(bad))
        if n_bad == 0:
            return {"certified": witness is None, "min_rel": best_rel,
                    "min_u": best_u, "witness": witness,
                    "unresolved": unresolved, "exhausted": False}
        if 2 * n_bad > max_boxes:
            break
        lo_b, hi_b = lo[bad], hi[bad]
        mid_b = (lo_b + hi_b) / 2
        lo = np.concatenate([lo_b, mid_b])
        hi = np.concatenate([mid_b, hi_b])

    return {"certified": False, "min_rel": best_rel, "min_u": best_u,
            "witness": witness, "unresolved": unresolved, "exhausted": True}


# ---------------------------------------------------------------------------
# Candidate interchange format (one schema for every producer and consumer)
# ---------------------------------------------------------------------------

CANDIDATE_FORMAT = "sign-uncertainty-candidate/1"


def _dec(x, digits: int = 21) -> str:
    if isinstance(x, str):
        return x
    return np.format_float_scientific(np.longdouble(x), precision=digits - 1,
                                      unique=False, trim="0")


def save_candidate(path, *, d: int, s: int, widths, coeffs, u0,
                   rho=None, precision_digits: int = 19,
                   source: str = "lp", taus=None, notes=None) -> dict:
    """
    Write a candidate in the one schema everything speaks.

    Numbers are decimal STRINGS, not JSON floats: the coefficients run to ~1e5
    against an O(1) function, so a float64 round trip perturbs f by ~1e-11 --
    the same size as the dips a verifier is meant to adjudicate.  Serialising
    through float64 silently changes the function being checked.
    """
    if rho is None:
        rho = math.sqrt(float(u0) / math.pi)
    doc = {
        "format": CANDIDATE_FORMAT,
        "d": int(d), "s": int(s),
        "widths": [_dec(a, precision_digits) for a in np.asarray(widths).ravel()],
        "coeffs": [_dec(c, precision_digits) for c in np.asarray(coeffs).ravel()],
        "u0": _dec(u0, precision_digits),
        "rho": _dec(rho, precision_digits),
        "precision_digits": int(precision_digits),
        "source": source,
    }
    if taus is not None:
        doc["taus"] = [_dec(t, precision_digits) for t in np.asarray(taus).ravel()]
    if notes:
        doc["notes"] = notes
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    return doc


def load_candidate(path) -> dict:
    """Read the schema; also accepts the pre-schema layout for regression files."""
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    if doc.get("format") != CANDIDATE_FORMAT:      # legacy fixtures
        doc = {"format": CANDIDATE_FORMAT,
               "d": doc.get("d", 1), "s": doc.get("s", 1),
               "widths": [repr(x) for x in doc["widths"]],
               "coeffs": [repr(x) for x in doc["coeffs"]],
               "u0": repr(doc["u0"]), "rho": repr(doc.get("rho", 0.0)),
               "precision_digits": 17, "source": "legacy-float64"}
    return doc


# ---------------------------------------------------------------------------
# LP: is there an admissible f that is nonnegative on [u0, oo)?
# ---------------------------------------------------------------------------

def positivity_nodes(u0: float, u_max: float, n: int) -> np.ndarray:
    """
    Nodes on [u0, u_max].  The function is a sum of exponentials with rates
    spanning [1/a_max, a_max], and e^{-a u} varies on the scale u ~ 1/a, so a
    LOG-spaced grid is the one that resolves every width equally.  We add a
    linear refinement just above u0, where the contact points live.
    """
    lo = max(u0, u_max * 1e-14)
    g = [np.geomspace(lo, u_max, n)]
    t = np.linspace(0.0, 1.0, max(n // 2, 8))
    near = min(u_max, u0 + max(5.0, 3.0 * u0))
    g.append(u0 + (near - u0) * t)
    g.append(u0 + (near - u0) * t ** 3)
    g.append(np.array([u0, u_max]))
    return np.unique(np.concatenate(g))


def _cleanup(fam: Family, coeffs: np.ndarray, uv: np.ndarray,
             drop: float = 1e-11) -> np.ndarray:
    """
    Zero out widths whose whole contribution is below the cancellation noise.

    The LP routinely returns a solution in which the widest Gaussian carries a
    coefficient of size ~1e-9 -- i.e. zero to within the simplex's own
    feasibility tolerance, but with an arbitrary SIGN.  Since that sign is what
    decides whether f is eventually positive or eventually negative, leaving it
    in makes the answer a coin flip.  Rounding it to zero moves the solution
    into the family with one fewer width, where the question is well posed.
    """
    c = np.array(coeffs, dtype=float)
    B = np.abs(fam.basis(uv)) * np.abs(c)[None, :]
    mag = np.maximum(fam.term_magnitude(uv, c), 1e-300)
    ratio = np.max(B / mag[:, None], axis=0)
    c[ratio < drop] = 0.0
    return c


XP = np.longdouble          # 64-bit mantissa: ~3 extra decimal digits


def _basis_xp(fam: Family, u) -> np.ndarray:
    """
    Rescaled basis in extended precision.

    All the ill-conditioning of this problem lives in the change of basis from
    the phi_j to a well-behaved one; once that change of basis has been done
    accurately, everything downstream is benign.  So we form phi and the
    orthogonalisation in np.longdouble (64-bit mantissa, ~19 digits) and hand
    float64 to the LP.  Cheap, and it buys roughly three orders of magnitude of
    coefficient growth -- empirically k = 16 -> k = 24.
    """
    u = np.atleast_1d(np.asarray(u, dtype=XP))
    a = np.asarray(fam.scales, dtype=XP)
    nu = XP(1.0) / a[-1]
    logw = -XP(0.5) * XP(fam.d) * np.log(a)
    head = np.exp(-np.outer(u, a - nu))
    tail = XP(fam.s) * np.exp(logw[None, :] - np.outer(u, XP(1.0) / a - nu))
    return head + tail


def _mgs(B: np.ndarray, rcond: float):
    """
    Column-pivoted modified Gram-Schmidt with reorthogonalisation, in
    longdouble.  numpy.linalg has no extended-precision path (LAPACK silently
    demotes to float64), so this is done by hand.

    Pivoting matters a lot here: the columns have wildly different norms on a
    grid that reaches into the tail (phi_1 decays like e^{-(1-nu)u} while the
    widest one tends to a constant), and unpivoted MGS starting from the
    smallest column loses most of the digits the longdouble was bought for.

    Returns T with B @ T orthonormal, dropping directions below rcond.
    """
    n, k = B.shape
    Q = np.zeros((n, k), dtype=XP)
    T = np.zeros((k, k), dtype=XP)
    R = np.zeros((k, k), dtype=XP)       # residual coefficient vectors
    V = B.copy()
    for j in range(k):
        R[j, j] = XP(1.0)
    used = np.zeros(k, dtype=bool)
    cols = 0
    scale0 = None
    for _ in range(k):
        nrms = np.array([np.sqrt(V[:, j] @ V[:, j]) if not used[j] else XP(0.0)
                         for j in range(k)], dtype=XP)
        j = int(np.argmax(nrms))
        if scale0 is None:
            scale0 = nrms[j]
        if nrms[j] <= rcond * scale0:
            break
        used[j] = True
        Q[:, cols] = V[:, j] / nrms[j]
        T[:, cols] = R[:, j] / nrms[j]
        q, t = Q[:, cols], T[:, cols]
        for jj in range(k):
            if used[jj]:
                continue
            for _pass in range(2):
                pr = q @ V[:, jj]
                V[:, jj] -= pr * q
                R[:, jj] -= pr * t
        cols += 1
    return T[:, :cols], cols


_PRECOND_CACHE: dict = {}


def _preconditioner(fam: Family, u0: float, u_max: float, n: int = 600,
                    rcond: float = 1e-11):
    """
    The raw columns phi_{a_j} are catastrophically ill-conditioned once the
    widths spread out (e^{-a u} with huge a is a spike at 0; e^{-u/a} with
    huge a is nearly constant).  Build an orthonormal basis of the SAME span
    on a reference grid: psi = phi @ T with T from an economy QR.  The
    feasible cone is unchanged, the LP becomes well scaled.
    Returns (T, kept_rank).
    """
    key = (fam.d, fam.s, tuple(fam.scales), round(u0, 12), round(u_max, 9), n, rcond)
    hit = _PRECOND_CACHE.get(key)
    if hit is not None:
        return hit
    ref = np.unique(np.concatenate([
        positivity_nodes(u0, u_max, n), np.array([0.0])]))
    B = _basis_xp(fam, ref)
    B = B / np.max(np.abs(B))
    T, rank = _mgs(B, rcond)
    out = (T * XP(math.sqrt(len(ref))), rank)
    if len(_PRECOND_CACHE) > 512:
        _PRECOND_CACHE.clear()
    _PRECOND_CACHE[key] = out
    return out


def _lp_leading(fam: Family, u0: float, n_grid: int, n_verify: int,
                tol_pos: float, max_cuts: int, max_ext: int,
                margin_ref=None, delta: float = 0.0):
    """
    Feasibility with the LEADING term normalised: the slowest-decaying term of
    f is pinned to coefficient +1.

    This is the fix for the one genuinely nasty numerical issue in the whole
    problem.  Near the optimal rho the extremal f wants the widest Gaussian's
    tail coefficient to vanish, so an unnormalised LP returns it at ~1e-9 --
    zero to within the simplex's feasibility tolerance, but with an ARBITRARY
    SIGN, and that sign alone decides whether f is eventually positive or
    eventually negative.  Pinning it to +1 removes the ambiguity; the caller
    then sweeps which width is the leading one, which covers the degenerate
    cases exactly rather than approximately.
    """
    gap = (1.0 / fam.scales[-2] - fam.nu) if fam.k >= 2 else 1.0
    u_max = max(8.0 * max(u0, 1.0), 25.0, u0 + (40.0 / gap if gap > 0 else 0.0))

    lead = np.zeros(fam.k)
    lead[-1] = fam.s * math.exp(fam.log_w[-1])      # gamma_0 as a functional of c

    for _ext in range(max_ext + 1):
        T, rank = _preconditioner(fam, u0, u_max, n=min(n_grid, 600))
        if rank < 1:
            return None
        nodes = positivity_nodes(u0, u_max, n_grid)
        A_eq = np.asarray(np.vstack([_basis_xp(fam, 0.0)[0] @ T,
                                      np.asarray(lead, dtype=XP) @ T]), dtype=float)
        b_eq = np.array([0.0, 1.0])
        sol = None
        for _cut in range(max_cuts + 1):
            Bn = np.asarray(_basis_xp(fam, nodes) @ T, dtype=float)
            rhs = (np.zeros(len(nodes)) if margin_ref is None
                   else -delta * np.asarray(margin_ref(nodes), dtype=float))
            res = linprog(c=np.zeros(T.shape[1]),
                          A_ub=-Bn,
                          b_ub=rhs,
                          A_eq=A_eq, b_eq=b_eq,
                          bounds=[(-1e9, 1e9)] * T.shape[1], method="highs")
            if not res.success:
                return None
            coeffs_xp = T @ np.asarray(res.x, dtype=XP)
            coeffs = np.asarray(coeffs_xp, dtype=float)
            uv = positivity_nodes(u0, u_max, n_verify)
            # evaluate through the orthonormalised basis: psi has O(1) columns
            # and x is O(1), so this does NOT suffer the cancellation that
            # sum_j c_j phi_j(u) does with |c| ~ 1e5.
            vals = np.asarray(_basis_xp(fam, uv) @ coeffs_xp, dtype=float)
            mag = np.maximum(fam.term_magnitude(uv, coeffs), 1e-300)
            rel = vals / mag           # dip measured in cancellation units
            if margin_ref is not None:
                rel = (vals - delta * np.asarray(margin_ref(uv))) / mag
            i = int(np.argmin(rel))
            if rel[i] >= -tol_pos:
                # the grid is only a cheap pre-filter; the authority is the
                # grid-free bound, which is what catches a split double root
                bb = certified_min_xp(fam, coeffs_xp, u0, u_max,
                                      tol_rel=tol_pos)
                if bb["witness"] is not None:
                    nodes = np.unique(np.concatenate(
                        [nodes, np.array([bb["witness"], bb["min_u"]])]))
                    continue
                sol = {"coeffs": coeffs, "coeffs_xp": coeffs_xp,
                       "bb": bb,
                       "T": T, "x": np.asarray(res.x, dtype=XP),
                       "u0": u0, "u_max": u_max, "rank": rank,
                       "min_grid": float(rel[i]), "min_grid_u": float(uv[i]),
                       "n_nodes": len(nodes)}
                break
            # Cut at the TRUE local minimisers, not at neighbouring grid
            # points.  The contact points of the optimal f are tangencies, so
            # grid-point cuts converge linearly and stall around 1e-8; a Brent
            # refinement of each violating dip converges in a handful of cuts.
            bad = np.where(rel < -tol_pos)[0]
            loc = bad[(bad > 0) & (bad < len(uv) - 1)]
            loc = loc[(rel[loc] <= rel[loc - 1]) & (rel[loc] <= rel[loc + 1])]
            cand = np.unique(np.concatenate([[i], loc]).astype(int))

            def _rel(t):
                v = float((_basis_xp(fam, t) @ coeffs_xp)[0])
                m = max(float(fam.term_magnitude(t, coeffs)[0]), 1e-300)
                return v / m

            extra = [uv[bad[:1]], uv[bad[-1:]]]
            for j in cand[:40]:
                lo_j, hi_j = uv[max(j - 1, 0)], uv[min(j + 1, len(uv) - 1)]
                if hi_j > lo_j:
                    r = minimize_scalar(_rel, bounds=(lo_j, hi_j), method="bounded",
                                        options={"xatol": 1e-13 * max(hi_j, 1.0)})
                    extra.append(np.array([r.x]))
                extra.append(uv[[j]])
            nodes = np.unique(np.concatenate([nodes] + extra))
        if sol is None:
            return None
        u_dom, lead_rel, verdict = fam.tail_dominance(sol["coeffs"])
        sol.update(u_dominance=u_dom, lead_rel=lead_rel, tail_verdict=verdict)
        if verdict != "ok":
            return None
        if u_dom <= u_max:
            sol["tail_ok"] = True
            return sol
        u_max = 1.6 * u_dom
    return None


_M_HINT: dict = {}


def lp_feasible(fam: Family, u0: float, n_grid: int = 500, n_verify: int = 8000,
                tol_pos: float = 3e-11, max_cuts: int = 20, max_ext: int = 4,
                margin_ref=None, delta: float = 0.0):
    """
    Is there an admissible f, nonnegative on [u0, oo)?  Sweep which width
    carries the slowest surviving term; the sub-family with widths a_1..a_m is
    exactly the set of solutions whose widest m..k coefficients vanish, so the
    sweep covers every degenerate configuration without ever asking floating
    point to adjudicate a coefficient of size 1e-9.
    """
    key = (fam.d, fam.s, tuple(fam.scales))
    order = list(range(fam.k, 1, -1))
    hint = _M_HINT.get(key)
    if hint in order:
        order.remove(hint); order.insert(0, hint)
    for m in order:
        try:
            sub = Family(fam.d, fam.s, fam.scales[:m])
        except ValueError:
            continue
        r = _lp_leading(sub, u0, n_grid, n_verify, tol_pos, max_cuts, max_ext,
                        margin_ref=margin_ref, delta=delta)
        if r is not None:
            c = np.zeros(fam.k)
            c[:m] = r["coeffs"]
            r["coeffs"] = c
            cx = np.zeros(fam.k, dtype=XP)
            cx[:m] = r["coeffs_xp"]
            r["coeffs_xp"] = cx
            r["m_leading"] = m
            r["family"] = sub
            _M_HINT[key] = m
            return r
    return None


def lp_feasible_robust(fam: Family, u0: float, n_grid: int = 400,
                       n_verify: int = 6000, **lp_kw):
    """
    Feasibility of the cone is monotone in u0, so a single spurious 'no'
    permanently derails the bisection.  Confirm every negative answer once
    on a finer grid before believing it.
    """
    sol = lp_feasible(fam, u0, n_grid=n_grid, n_verify=n_verify, **lp_kw)
    if sol is not None:
        return sol
    return lp_feasible(fam, u0, n_grid=2 * n_grid, n_verify=3 * n_verify,
                       max_cuts=lp_kw.pop("max_cuts", 14), **lp_kw)


def min_rho(fam: Family, bisect: int = 40, u_hi: float | None = None, **lp_kw):
    """Smallest u0 (hence rho = sqrt(u0/pi)) for which the LP is feasible."""
    if u_hi is None:
        u_hi = 4.0 * math.pi * bck_upper(fam.d) ** 2
    best = lp_feasible_robust(fam, u_hi, **lp_kw)
    if best is None:
        return None
    lo, hi = 0.0, u_hi  # u0 = 0 is always infeasible (it forces f == 0)
    for _ in range(bisect):
        mid = 0.5 * (lo + hi)
        sol = lp_feasible_robust(fam, mid, **lp_kw)
        if sol is None:
            lo = mid
        else:
            hi, best = mid, sol
    best["rho"] = math.sqrt(best["u0"] / math.pi)
    return best


def safe_rho(fam: Family, delta: float = 1e-9, **kw):
    """
    Two-pass rho with a positive margin.

    Pass 1 is the ordinary search; its solution supplies a fixed positive
    envelope E(u) = sum_i |gamma_i| e^{-(beta_i - nu) u}.  Pass 2 re-runs the
    whole bisection demanding f >= delta * E on the ray.  The result is a
    number that survives an independent high-precision replay -- pass-1 output
    at large k typically does not, because the LP is then working entirely
    inside the cancellation noise and its "nonnegative" solutions dip by
    ~1e-12 of E.  Costs one extra search; report both.
    """
    raw = min_rho(fam, **kw)
    if raw is None:
        return None, None
    ref_coeffs = raw["coeffs"].copy()
    env = lambda u: fam.term_magnitude(u, ref_coeffs)
    safe = min_rho(fam, margin_ref=env, delta=delta, **kw)
    return raw, safe


# ---------------------------------------------------------------------------
# Root structure of a discovered solution
# ---------------------------------------------------------------------------

def analyse_roots(fam: Family, coeffs, u_hi: float, n=40000, tol_rel=1e-7):
    """Locate sign changes and near-double roots of f on (0, u_hi]."""
    u = np.unique(np.concatenate([np.linspace(1e-12, u_hi, n),
                                  np.geomspace(1e-8, u_hi, n)]))
    v = fam.eval(u, coeffs)
    scale = float(np.max(np.abs(v)))
    simple, double = [], []
    sgn = np.sign(v)
    idx = np.where(sgn[1:] * sgn[:-1] < 0)[0]
    for i in idx:
        try:
            simple.append(brentq(lambda t: fam.eval(t, coeffs)[0], u[i], u[i + 1]))
        except Exception:
            pass
    # interior local minima that nearly touch zero from above
    for i in range(1, len(u) - 1):
        if v[i] <= v[i - 1] and v[i] <= v[i + 1] and abs(v[i]) < tol_rel * scale:
            if not double or abs(u[i] - double[-1]) > 1e-4 * u_hi:
                double.append(float(u[i]))
    # a near-double root shows up as two nearby brentq hits; merge them
    if double:
        simple = [x for x in simple
                  if min(abs(x - y) for y in double) > 1e-3 * max(u_hi, 1.0)]
    return {
        "simple_roots_u": simple,
        "double_roots_u": double,
        "simple_roots_r": [math.sqrt(x / math.pi) for x in simple],
        "double_roots_r": [math.sqrt(x / math.pi) for x in double],
        "n_roots_with_mult": len(simple) + 2 * len(double),
        "descartes_bound": fam.descartes_bound(coeffs),
    }


# ---------------------------------------------------------------------------
# Width tuning: the degree of freedom polynomial x Gaussian does not have
# ---------------------------------------------------------------------------

def _widths_from_theta(theta: np.ndarray, s: int) -> np.ndarray:
    """theta -> strictly increasing widths >= 1 (a_1 = 1 fixed when s = +1)."""
    steps = np.exp(np.clip(theta, -8.0, 8.0))
    a = 1.0 + np.cumsum(steps)
    return np.concatenate([[1.0], a]) if s == 1 else a


def tune_widths(d: int, s: int, k: int, theta0=None, maxiter=200, verbose=True,
                **lp_kw):
    """Nelder-Mead on log-widths, minimising rho.  Coefficients from the LP."""
    n_free = k - 1 if s == 1 else k
    if theta0 is None:
        theta0 = np.log(np.full(n_free, 0.6) * (1.5 ** np.arange(n_free)))
    history = []

    def obj(theta):
        try:
            fam = Family(d, s, _widths_from_theta(np.asarray(theta), s))
            sol = min_rho(fam, bisect=26, **lp_kw)
        except Exception:
            return 10.0
        if sol is None:
            return 10.0
        history.append((float(sol["rho"]), fam.scales.copy()))
        if verbose and (len(history) % 20 == 0 or sol["rho"] <= min(h[0] for h in history)):
            print(f"    [{len(history):4d}] rho = {sol['rho']:.8f}  "
                  f"widths = {np.array2string(fam.scales, precision=4)}")
        return float(sol["rho"])

    res = minimize(obj, np.asarray(theta0, dtype=float), method="Nelder-Mead",
                   options={"maxiter": maxiter, "xatol": 1e-4, "fatol": 1e-9})
    best_rho, best_scales = min(history, key=lambda t: t[0]) if history else (None, None)
    fam = Family(d, s, best_scales)
    sol = min_rho(fam, bisect=44, **lp_kw)
    return fam, sol, res


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(fam: Family, sol: dict, roots: dict | None = None) -> None:
    d, s = fam.d, fam.s
    pub = published_upper(d, s)
    lb, lb_name = best_lower_bound(d, s)
    print("=" * 76)
    print(f"Gaussian-mixture eigenfunction search   d = {d}   s = {s:+d}   k = {fam.k}")
    print("=" * 76)
    print(f"widths a_j        : {np.array2string(fam.scales, precision=6)}")
    print(f"coefficients c_j  : {np.array2string(sol['coeffs'], precision=6)}")
    print(f"rho (candidate)   : {sol['rho']:.10f}")
    print(f"  u0 = pi rho^2   : {sol['u0']:.10f}")
    print(f"grid min (rel)    : {sol['min_grid']:+.3e} at u = {sol['min_grid_u']:.6f}")
    bb = sol.get("bb")
    if bb:
        print(f"grid-free bound   : {bb['min_rel']:+.3e} at u = {bb['min_u']:.9f}   "
              f"certified={bb['certified']} unresolved_boxes={bb['unresolved']}"
              + ("  (box budget exhausted)" if bb["exhausted"] else ""))
    mass = float(fam.mass_functional(sol["u0"]) @ sol["coeffs"])
    print(f"leading width     : a_{sol['m_leading']} = {fam.scales[sol['m_leading']-1]:.6f}"
          f"   (widths above it carry zero coefficient)")
    print(f"mass invariant    : {mass:+.6e}   "
          f"(= e^(nu u0) int_u0^oo f; must be > 0 since int_0^oo f = s f(0) = 0)")
    print(f"tail dominates from u = {sol['u_dominance']:.4g}  "
          f"(leading term = {sol['lead_rel']:.2e} of max, "
          f"verdict '{sol['tail_verdict']}')")
    if roots is not None:
        print(f"roots on the ray  : {len(roots['simple_roots_u'])} simple, "
              f"{len(roots['double_roots_u'])} double "
              f"({roots['n_roots_with_mult']} with multiplicity; "
              f"Descartes cap {roots['descartes_bound']})")
        if roots["double_roots_r"]:
            print("  double roots r  : "
                  + np.array2string(np.array(roots["double_roots_r"]), precision=5))
    print("-" * 76)
    print(f"best known LOWER  : {lb:.6f}   ({lb_name})")
    if pub is not None:
        tag = "EXACT" if (s, d) in EXACT_VALUES else "upper bd [CG] Tab 4.1"
        print(f"published A_{'+' if s == 1 else '-'}({d})   : {pub:.6f}   ({tag})")
        gap = sol["rho"] - pub
        print(f"candidate - published: {gap:+.6f}  "
              f"({'BEATS PUBLISHED -> CERTIFY' if gap < 0 else 'no improvement yet'})")
    if s == 1:
        print(f"[BCK] 3-Gaussian  : {bck_upper(d):.6f}  "
              f"(ratio {sol['rho'] / bck_upper(d):.4f})")
    print(f"[CDG] sqrt(d/2pi) : {cdg_asymptote(d):.6f}  "
          f"(rho * sqrt(2pi/d) = {sol['rho'] / cdg_asymptote(d):.6f})")
    print("-" * 76)
    print("DISCOVERY ONLY: positivity is grid-imposed; no certificate here.")


# ---------------------------------------------------------------------------
# Experiment drivers
# ---------------------------------------------------------------------------

def mode_single(args):
    scales = (np.array([float(v) for v in args.scales.split(",")])
              if args.scales else make_scales(args.k, args.sign, args.spread))
    fam = Family(args.d, args.sign, scales)
    sol = min_rho(fam, bisect=args.bisect)
    if sol is None:
        print("infeasible: no admissible f found in this family / bracket")
        return None, None
    roots = analyse_roots(fam, sol["coeffs"], min(sol["u_max"], 12 * sol["u0"] + 20))
    report(fam, sol, roots)
    return fam, sol


def mode_tune(args):
    print(f"tuning {args.k} widths for d = {args.d}, s = {args.sign:+d} ...")
    fam, sol, _ = tune_widths(args.d, args.sign, args.k, maxiter=args.maxiter)
    if sol is None:
        print("tuning failed")
        return None, None
    roots = analyse_roots(fam, sol["coeffs"], min(sol["u_max"], 12 * sol["u0"] + 20))
    report(fam, sol, roots)
    return fam, sol


def mode_kscan(args):
    """Does rho decrease with the number of widths, and how fast?"""
    pub = published_upper(args.d, args.sign)
    lb, _ = best_lower_bound(args.d, args.sign)
    print(f"k-scan   d = {args.d}   s = {args.sign:+d}   "
          f"(lower bd {lb:.6f}, published {pub if pub else float('nan'):.6f})")
    print(f"{'k':>3} {'rho(fixed grid)':>16} {'rho(tuned)':>14} "
          f"{'/published':>11} {'roots':>7} {'Descartes':>10}")
    rows = []
    for k in range(2, args.kmax + 1):
        fam0 = Family(args.d, args.sign, make_scales(k, args.sign, args.spread))
        s0 = min_rho(fam0, bisect=30)
        r0 = s0["rho"] if s0 else float("nan")
        fam, sol, _ = tune_widths(args.d, args.sign, k,
                                  maxiter=args.maxiter, verbose=False)
        if sol is None:
            print(f"{k:>3} {r0:>16.8f} {'--':>14}")
            continue
        rr = analyse_roots(fam, sol["coeffs"], min(sol["u_max"], 12 * sol["u0"] + 20))
        rel = sol["rho"] / pub if pub else float("nan")
        print(f"{k:>3} {r0:>16.8f} {sol['rho']:>14.8f} {rel:>11.5f} "
              f"{rr['n_roots_with_mult']:>7} {rr['descartes_bound']:>10}")
        rows.append({"k": k, "rho_fixed": r0, "rho_tuned": sol["rho"],
                     "widths": fam.scales.tolist(),
                     "roots": rr["n_roots_with_mult"],
                     "descartes": rr["descartes_bound"]})
    return rows


def mode_asymptotic(args):
    """
    THE decisive experiment.  [CDG] says degree-o(d) polynomials are stuck at
    rho ~ sqrt(d/2pi).  Track rho * sqrt(2pi/d) for fixed k as d grows.  If it
    stays >= 1, the Chebyshev-system obstruction has caught us and k must grow
    with d; if it dips below 1, this family genuinely escapes.
    """
    ds = [int(v) for v in args.dims.split(",")]
    print(f"asymptotic scan   s = {args.sign:+d}   k = {args.k}")
    print(f"{'d':>5} {'rho':>13} {'rho*sqrt(2pi/d)':>17} {'published':>11} "
          f"{'pub*sqrt(2pi/d)':>17} {'lower*sqrt(2pi/d)':>19}")
    rows = []
    for d in ds:
        fam, sol, _ = tune_widths(d, args.sign, args.k,
                                  maxiter=args.maxiter, verbose=False)
        if sol is None:
            print(f"{d:>5} {'--':>13}")
            continue
        norm = 1.0 / cdg_asymptote(d)
        pub = published_upper(d, args.sign)
        lb, _ = best_lower_bound(d, args.sign)
        print(f"{d:>5} {sol['rho']:>13.7f} {sol['rho'] * norm:>17.6f} "
              f"{(pub if pub else float('nan')):>11.6f} "
              f"{((pub * norm) if pub else float('nan')):>17.6f} "
              f"{lb * norm:>19.6f}")
        rows.append({"d": d, "rho": sol["rho"], "rho_norm": sol["rho"] * norm,
                     "widths": fam.scales.tolist()})
    return rows


def make_plot(fam: Family, sol: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    rho = sol["rho"]
    r = np.linspace(0.0, max(3.0, 2.5 * rho), 4000)
    f = fam.eval_r(r, sol["coeffs"])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(r, f, lw=1.3)
    ax[0].axhline(0, lw=0.7, color="k")
    ax[0].axvline(rho, ls="--", lw=1.0, color="C3", label=f"rho = {rho:.6f}")
    ax[0].set_xlabel("r"); ax[0].set_ylabel("f(r)")
    ax[0].set_title(f"d = {fam.d}, s = {fam.s:+d}, k = {fam.k}")
    ax[0].legend()
    mask = r >= rho * 0.98
    scale = np.max(np.abs(f)) or 1.0
    ax[1].plot(r[mask], f[mask] / scale, lw=1.2)
    ax[1].axhline(0, lw=0.7, color="k")
    ax[1].set_yscale("symlog", linthresh=1e-12)
    ax[1].set_xlabel("r"); ax[1].set_title("tail, symlog (double roots visible)")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("single", "tune", "kscan", "asymptotic"),
                    default="single")
    ap.add_argument("--d", type=int, default=12)
    ap.add_argument("--sign", type=int, choices=(-1, 1), default=1)
    ap.add_argument("--k", type=int, default=5, help="number of Gaussian widths")
    ap.add_argument("--scales", default=None,
                    help="explicit comma-separated widths (overrides --k)")
    ap.add_argument("--spread", type=float, default=1.6,
                    help="ratio for the default geometric widths")
    ap.add_argument("--bisect", type=int, default=40)
    ap.add_argument("--maxiter", type=int, default=160)
    ap.add_argument("--kmax", type=int, default=7)
    ap.add_argument("--dims", default="4,8,16,32,64")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--plot", type=Path, default=None)
    args = ap.parse_args()

    out = None
    if args.mode == "single":
        fam, sol = mode_single(args)
    elif args.mode == "tune":
        fam, sol = mode_tune(args)
    elif args.mode == "kscan":
        fam = sol = None
        out = mode_kscan(args)
    else:
        fam = sol = None
        out = mode_asymptotic(args)

    if args.json:
        if out is not None:
            args.json.write_text(json.dumps(out, indent=2), encoding="utf-8")
        elif sol is not None:
            save_candidate(args.json, d=fam.d, s=fam.s, widths=fam.scales,
                           coeffs=sol["coeffs_xp"], u0=sol["u0"],
                           rho=sol["rho"], precision_digits=19, source="lp")
        print(f"JSON written to {args.json}")
    if args.plot and fam is not None and sol is not None:
        make_plot(fam, sol, args.plot)
        print(f"plot written to {args.plot}")


if __name__ == "__main__":
    main()
