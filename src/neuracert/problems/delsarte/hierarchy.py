"""DelsarteLin(r, n, d): the Loyfer-Linial / CJJ higher-order Delsarte LP,
symmetrized, with exact multivariate Krawtchouk tables and exact dual
certification.  Implemented against LL22 (arXiv:2206.09211) Definitions 3-5
and Propositions 7-8, and CJJ+25 (arXiv:2501.04854) Section 2.

THE OBJECTS
  Configurations I_{r,n}: alpha: F_2^r -> N with sum alpha = n -- the column
  enumerator of X in {0,1}^{r x n} (LL Def 4).  |I_{r,n}| = C(n+2^r-1, 2^r-1).

  Row-span weights: |u^T X| = sum_{v: <u,v>=1} alpha_v  (LL Prop 2, in the
  form that avoids Fourier entirely).  X is Valid for distance d iff every
  u != 0 has |u^T X| = 0 or >= d.

  GL(r,2) acts by (T.alpha)_u = alpha_{T^{-1} u}.  For r <= 2 the action on
  the 2^r - 1 nonzero u's is the FULL symmetric group, which is why sorting
  the nonzero part used to be a legitimate canonical form.  For r >= 3 it is
  not: |GL(3,2)| = 168 while |S_7| = 5040, so sorting over-merges by a factor
  30 and the Fano incidence structure (GL(3,2) = PSL(2,7) acting on the seven
  points of PG(2,2)) is exactly what distinguishes the orbits.  The group is
  therefore enumerated explicitly and orbits are computed by group action, at
  a cost of one group pass per orbit -- O(|I_{r,n}|) overall, not
  O(|GL| |I_{r,n}|).  Every r is supported; the practical wall is
  |I_{r,n}| = C(n + 2^r - 1, 2^r - 1), not the group.

  PARTIAL TRANSFORMS FOR GENERAL S (the actual obstruction at r >= 3).
  For S a set of rows, X must agree with Y off S, so the columns split into
  the cosets of F_2^{S^c} and the sum factors:

      K^S_alpha(beta) = prod_{w in F_2^{S^c}} K^{(|S|)}_{alpha(.,w)}(beta(.,w))

  vanishing unless the S^c-marginals of alpha and beta agree.  Each factor is
  a FULL |S|-dimensional Krawtchouk, so the same machinery recurses: at r = 3
  the |S| = 1 family is a product of four classical Krawtchouks and the
  |S| = 2 family a product of two bivariate ones.  LL Prop 8 is the r = 2,
  |S| = 1 case.

  RECIPROCITY AND THE TRANSPOSED ROW BUILD.  Counting pairs (X, Y) of types
  (alpha, beta) weighted by (-1)^{<X,Y>} in the two possible orders gives

      multinom(beta) K_alpha(beta) = multinom(alpha) K_beta(alpha)

  and combining it with GL-equivariance collapses the sum over an alpha-orbit
  into an orbit sum of a SINGLE expansion:

      sum_{alpha in O_j} K_alpha(beta)
          = multinom(a_j) |O_j| / (multinom(beta) |O(beta)|)
            * sum_{beta' in O(beta)} K_{beta'}(a_j).

  Building the table this way costs one expansion per VALID orbit instead of
  one per beta orbit.  Since the valid orbits are a subset of all orbits this
  is never worse and is usually much better -- at r = 3, n = 14, d = 6 it is
  112 expansions instead of 1235.  Both routes are cross-checked against each
  other in the self-test.

  Multivariate Krawtchouk (LL Prop 7, generating-function form):

      K_alpha(beta) = [w^alpha]  prod_u ( sum_v (-1)^{<u,v>} w_v )^{beta_u}

  computed exactly by sparse polynomial multiplication; one expansion gives
  the whole row K_.(beta) at once.

  Partial Krawtchouks (LL Prop 8): K^S_alpha(beta) factors into a product
  of order-|S| Krawtchouks over the F_2^{r-|S|} groups, and vanishes unless
  the group marginals of alpha and beta agree.  For r = 2 and |S| = 1 these
  are products of two classical binary Krawtchouks.

THE LP (LL Def 5, q = 2, so the beta variables of CJJ (2) drop since X = -X)
  variables phi_O >= 0 per valid GL-orbit O          (S = empty constraint)
  phi_{orbit of n eps_0} = 1                          (Normalization)
  sum_O phi_O K^S(beta, O) >= 0   for S in {{1},{2},[r]}, all beta  (C2)
  objective (Obj):  1 + sum_{k valid} C(n,k) phi_{orbit(n-k, k)}
            (Obj'): sum_O mass(O) phi_O,  mass(O) = sum_{a in O} multinom(a)

  (Obj) upper-bounds A^Lin(n,d); (Obj') upper-bounds A^Lin(n,d)^r.

EXACT CERTIFICATION AND THE REPAIR IDENTITY
  Weak duality: any y >= 0 with (B^T y)_O <= -c_O for every valid orbit
  O != O_0 certifies  A^Lin <= c_{O_0} + (B^T y)_{O_0}  (B = constraint
  rows).  The repair direction generalizing the level-1 reference dual is

      sum_{beta in I_{r,n}} multinom(beta) K_alpha(beta) = 2^{rn} [alpha = n eps_0]

  (Fourier inversion at 0: summing L_hat over ALL X kills everything except
  L_alpha(0)).  Hence y_hat with y_hat(beta) = multinom(beta) on beta != n eps_0
  gives (B^T y_hat)_alpha = -multinom(alpha) < 0 strictly on every alpha,
  and 2^{rn} - 1 at the normalization orbit: a total exact repair at the
  quantified cost mu (2^{rn} - 1), exactly as at level 1.  The identity is
  VERIFIED against the computed table before any certificate is issued --
  it is simultaneously the global correctness check of the whole table,
  playing the role P Q = |X| I played for association schemes.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache


# ---------------------------------------------------------------------------
# Configurations, weights, validity and exact GL(r,2) orbits
# ---------------------------------------------------------------------------
def _check_r(r: int) -> None:
    if r < 1:
        raise ValueError(f"r = {r} must be >= 1")
    if r > 4:
        raise NotImplementedError(
            f"r = {r}: |I_{{r,n}}| = C(n + {(1 << r) - 1}, {(1 << r) - 1}) is "
            f"astronomically large for any useful n; the orbit machinery is "
            f"correct but the LP is not buildable. Refusing.")


@lru_cache(maxsize=None)
def gl_group(r: int) -> tuple:
    """GL(r,2) as permutations of F_2^r: p[v] = T v, indexed by v as an int.

    |GL(r,2)| = 1, 6, 168, 20160 for r = 1, 2, 3, 4.  The orbit routines cost
    one group pass per ORBIT, so the group size cancels against the orbit
    size and the total is O(|I_{r,n}|).
    """
    from itertools import product as _product
    m = 1 << r
    out = []
    for cols in _product(range(1, m), repeat=r):
        span = {0}
        for c in cols:
            span |= {x ^ c for x in span}
        if len(span) != m:              # columns must span, i.e. T invertible
            continue
        perm = [0] * m
        for v in range(m):
            img = 0
            for i in range(r):
                if (v >> i) & 1:
                    img ^= cols[i]
            perm[v] = img
        out.append(tuple(perm))
    return tuple(out)


def orbit_map(cfgs, r: int):
    """(reps, rep_of, orbit_size) for the GL(r,2) action on a config list."""
    m = 1 << r
    G = gl_group(r)
    rep_of, reps, size = {}, [], {}
    for a in cfgs:
        if a in rep_of:
            continue
        orb = {tuple(a[p[v]] for v in range(m)) for p in G}
        rep = min(orb)
        for b in orb:
            rep_of[b] = rep
        reps.append(rep)
        size[rep] = len(orb)
    return reps, rep_of, size


def all_configs(n: int, r: int):
    """All alpha in I_{r,n}, as tuples of length 2^r."""
    m = 1 << r
    out = []

    def rec(prefix, left, slots):
        if slots == 1:
            out.append(prefix + (left,))
            return
        for v in range(left + 1):
            rec(prefix + (v,), left - v, slots - 1)

    rec((), n, m)
    return out


def canon(alpha, r: int):
    """Canonical GL(r,2)-orbit representative: lexicographic minimum image."""
    m = 1 << r
    return min(tuple(alpha[p[v]] for v in range(m)) for p in gl_group(r))


def orbit_members(rep, r: int):
    """All configurations in the GL(r,2)-orbit of `rep`."""
    m = 1 << r
    return sorted({tuple(rep[p[v]] for v in range(m)) for p in gl_group(r)})


def weights(alpha, r: int):
    """(w_u)_{u != 0} with w_u = |u^T X| = sum_{v: <u,v> = 1} alpha_v."""
    m = 1 << r
    return tuple(sum(alpha[v] for v in range(m) if bin(u & v).count("1") & 1)
                 for u in range(1, m))


def is_valid(alpha, d: int, r: int, even: bool) -> bool:
    """Membership of Valid_{n,r}: every row-span weight is 0 or >= d.

    `even` additionally requires every weight even -- the standard reduction
    for even d (an even-weight code attains the optimum), which LL apply in
    their numerical section; kept as a flag so both conventions can be run.
    """
    for w in weights(alpha, r):
        if 0 < w < d:
            return False
        if even and (w & 1):
            return False
    return True


def multinom(alpha) -> int:
    out, tot = 1, 0
    for a in alpha:
        tot += a
        out *= math.comb(tot, a)
    return out


# ---------------------------------------------------------------------------
# Exact Krawtchouk tables
# ---------------------------------------------------------------------------
def kraw_row(beta, r: int) -> dict:
    """{alpha: K_alpha(beta)} for ALL alpha at once, exactly.

    Expand prod_u L_u^{beta_u} with L_u = sum_v (-1)^{<u,v>} w_v by repeated
    sparse multiplication; the coefficient of w^alpha is K_alpha(beta).
    """
    m = 1 << r
    signs = [[1 - 2 * (bin(u & v).count("1") & 1) for v in range(m)]
             for u in range(m)]
    unit = [0] * m
    poly = {tuple(unit): 1}
    for u in range(m):
        for _ in range(beta[u]):
            nxt: dict = {}
            su = signs[u]
            for exp, c in poly.items():
                for v in range(m):
                    e2 = list(exp)
                    e2[v] += 1
                    e2 = tuple(e2)
                    nxt[e2] = nxt.get(e2, 0) + su[v] * c
            poly = nxt
    return poly


def _kraw_mul(poly, su, m):
    nxt = {}
    get = nxt.get
    for exp, coeff in poly.items():
        for v in range(m):
            e2 = exp[:v] + (exp[v] + 1,) + exp[v + 1:]
            nxt[e2] = get(e2, 0) + su[v] * coeff
    return nxt


def _kraw_pow(poly, su, m, k):
    for _ in range(k):
        poly = _kraw_mul(poly, su, m)
    return poly


def kraw_rows(n: int, emit, r: int = 2) -> None:
    """emit(beta_rep, {alpha: K_alpha(beta)}) over every GL-orbit rep, exactly.

    kraw_row rebuilds prod_u L_u^{beta_u} from scratch for each beta, costing
    n sparse multiplications per row.  The beta's form a lattice and the
    factors commute, so a prefix tree shares the work: everything with the
    same (b0, b3, b2) prefix is computed once.  Because canonical reps satisfy
    b1 <= b2 <= b3, nesting b0 -> b3 -> b2 -> b1 puts the UNSHARED innermost
    level at cost min(b1,b2,b3) instead of n.  Same integers, no floats; about
    2x wall-clock over the whole table at n = 16..24 (the shared factors sit
    at high degree, so the ~8x reduction in multiplication COUNT does not
    translate one-for-one).
    """
    if r != 2:
        raise NotImplementedError("kraw_rows is specialised to r = 2")
    m = 4
    S = [[1 - 2 * (bin(u & v).count("1") & 1) for v in range(m)]
         for u in range(m)]
    P0 = {(0, 0, 0, 0): 1}
    for b0 in range(n + 1):
        if b0:
            P0 = _kraw_mul(P0, S[0], m)
        R0 = n - b0
        lo3 = -(-R0 // 3)
        P3 = _kraw_pow(P0, S[3], m, lo3)
        for b3 in range(lo3, R0 + 1):
            if b3 > lo3:
                P3 = _kraw_mul(P3, S[3], m)
            R = R0 - b3
            lo2, hi2 = -(-R // 2), min(b3, R)
            if lo2 > hi2:
                continue
            P2 = _kraw_pow(P3, S[2], m, lo2)
            for b2 in range(lo2, hi2 + 1):
                if b2 > lo2:
                    P2 = _kraw_mul(P2, S[2], m)
                emit((b0, R - b2, b2, b3), _kraw_pow(P2, S[1], m, R - b2))


@lru_cache(maxsize=None)
def kraw1(a1: int, b1: int, m: int) -> int:
    """Classical binary Krawtchouk: [w0^{m-a1} w1^{a1}] (w0+w1)^{m-b1}(w0-w1)^{b1}."""
    return sum((-1) ** j * math.comb(b1, j) * math.comb(m - b1, a1 - j)
               for j in range(0, min(a1, b1) + 1))


@lru_cache(maxsize=None)
def _sub_index(S: int, r: int):
    """(compress_S, compress_Sc, |S|) for the row bitmask S."""
    sbits = [i for i in range(r) if (S >> i) & 1]
    cbits = [i for i in range(r) if not (S >> i) & 1]
    m = 1 << r
    cs = tuple(sum(((v >> b) & 1) << k for k, b in enumerate(sbits))
               for v in range(m))
    cc = tuple(sum(((v >> b) & 1) << k for k, b in enumerate(cbits))
               for v in range(m))
    return cs, cc, len(sbits)


@lru_cache(maxsize=None)
def _sub_kraw(beta_sub: tuple, s_dim: int):
    """Memoised full |S|-dimensional Krawtchouk row for a coset."""
    return kraw_row(beta_sub, s_dim)


def _coset_split(alpha, cs, cc, ncos, nsub, m):
    """alpha as a list of per-coset sub-configurations on F_2^S."""
    out = [[0] * nsub for _ in range(ncos)]
    for v in range(m):
        out[cc[v]][cs[v]] += alpha[v]
    return out


def marginal(alpha, S: int, r: int) -> tuple:
    """The S^c-marginal of alpha: column mass per coset of F_2^{S^c}."""
    _, cc, s_dim = _sub_index(S, r)
    out = [0] * (1 << (r - s_dim))
    for v in range(1 << r):
        out[cc[v]] += alpha[v]
    return tuple(out)


def partial_k(alpha, beta, S: int, r: int) -> int:
    """K^S_alpha(beta), S a BITMASK of transformed rows (bit i = row i+1).

    X has to agree with Y off S, so the n columns split into the cosets of
    F_2^{S^c} and the character sum factors over them; inside a coset it is a
    full |S|-dimensional Krawtchouk on the sub-configurations.  Zero unless
    the S^c-marginals of alpha and beta agree.  For r = 2 and |S| = 1 this is
    LL Prop 8 and the bitmask coincides with the old row index (S = 1, 2).
    """
    cs, cc, s_dim = _sub_index(S, r)
    ncos = 1 << (r - s_dim)
    nsub = 1 << s_dim
    A = [[0] * nsub for _ in range(ncos)]
    B = [[0] * nsub for _ in range(ncos)]
    for v in range(1 << r):
        A[cc[v]][cs[v]] += alpha[v]
        B[cc[v]][cs[v]] += beta[v]
    out = 1
    for w in range(ncos):
        na, nb = sum(A[w]), sum(B[w])
        if na != nb:
            return 0
        if s_dim == 1:
            out *= kraw1(A[w][1], B[w][1], na)
        else:
            out *= _sub_kraw(tuple(B[w]), s_dim).get(tuple(A[w]), 0)
        if out == 0:
            return 0
    return out


# ---------------------------------------------------------------------------
# Brute-force cross-check (independent implementation, tiny n)
# ---------------------------------------------------------------------------
def brute_check(n: int = 3, r: int = 2) -> bool:
    """Verify kraw_row and partial_k against the DEFINITION, any r.

    Enumerates every X in {0,1}^{2 x n}, computes the (partial) Fourier
    transform of every level-set indicator directly from the character sum,
    and compares with the generating-function and product-formula values.
    This is the anchor that does not share a formula with the fast path.
    """
    m = 1 << r
    mats = []          # X as a tuple of n column-values in 0..m-1

    def rec(p):
        if len(p) == n:
            mats.append(tuple(p))
            return
        for v in range(m):
            rec(p + [v])

    rec([])
    cfgs = all_configs(n, r)
    gamma = {X: tuple(X.count(v) for v in range(m)) for X in mats}

    def ip(X, Y, rows):        # sum over selected rows of <X_row, Y_row>
        s = 0
        for j in range(n):
            for b in rows:
                s += ((X[j] >> b) & 1) * ((Y[j] >> b) & 1)
        return s & 1

    allrows = tuple(range(r))
    for alpha in cfgs:
        La = {X: 1 if gamma[X] == alpha else 0 for X in mats}
        # full transform: K_alpha(Gamma_X) = sum_Y (-1)^{<X,Y>} L_alpha(Y)
        for X in mats:
            full = sum((1 - 2 * ip(X, Y, allrows)) * La[Y] for Y in mats)
            if full != kraw_row(gamma[X], r).get(alpha, 0):
                return False
        # every proper nonempty row subset S: Y must agree with X off S
        for S in range(1, m - 1):
            rows = tuple(i for i in range(r) if (S >> i) & 1)
            off = tuple(i for i in range(r) if not (S >> i) & 1)
            for X in mats:
                acc = 0
                for Y in mats:
                    if all(((Y[j] >> b) & 1) == ((X[j] >> b) & 1)
                           for j in range(n) for b in off):
                        acc += (1 - 2 * ip(X, Y, rows)) * La[Y]
                if acc != partial_k(alpha, gamma[X], S, r):
                    return False
    return True


# ---------------------------------------------------------------------------
# The LP
# ---------------------------------------------------------------------------
@dataclass
class HierarchyLP:
    n: int
    d: int
    r: int
    even: bool
    valid_orbits: list          # canonical reps, index 0 = normalization
    orbit_mass: list            # sum of multinomials over the orbit
    rows: list                  # integer coefficient vectors over valid orbits
    row_kind: list              # ("full", beta_rep) or ("S1"/"S2", beta)
    c_obj: list                 # (Obj) coefficients
    c_objp: list                # (Obj') coefficients


def build_lp(n: int, d: int, r: int = 2, even: bool | None = None,
             verbose: bool = False) -> HierarchyLP:
    _check_r(r)
    if even is None:
        even = False
    # NOTE ON THE EVEN TRICK.  Zeroing variables in this LP does not merely
    # restrict the primal: it removes terms from the >= 0 constraint rows,
    # which can RELAX them and RAISE the value ((17,8): 32 -> 33.9).  So the
    # even restriction is not free here, and reproducing LL's published
    # table requires even=False in every tested case.  The flag stays for
    # experiments, off by default, and any certificate records it.
    m = 1 << r

    cfgs = all_configs(n, r)
    reps, rep_of, size = orbit_map(cfgs, r)
    zero_cfg = tuple([n] + [0] * (m - 1))
    valid_reps = [a for a in reps if is_valid(a, d, r, even)]
    valid_reps.sort(key=lambda a: (a != zero_cfg, a))
    if not valid_reps or valid_reps[0] != zero_cfg:
        raise RuntimeError("normalization configuration missing")
    col = {a: i for i, a in enumerate(valid_reps)}
    # multinom is constant on a GL-orbit (the group permutes coordinates), so
    # the orbit mass is just |O| * multinom(rep).
    mass = [size[a] * multinom(a) for a in valid_reps]

    # -- full-Krawtchouk rows, one per beta-ORBIT.  sum_{alpha in O} K_alpha
    #    is constant on the beta-orbit by GL-equivariance (K_{T a}(T b) =
    #    K_a(b) reindexes the sum), so one representative per orbit carries
    #    the whole family of constraints.  This is NOT asserted at runtime;
    #    the multinomial identity check in certify() is the global table
    #    check, and selftest exercises the invariance directly.
    #
    #    Built through reciprocity: one expansion per VALID orbit rather than
    #    one per beta orbit.  See the module docstring.
    beta_reps = reps
    bidx = {b: i for i, b in enumerate(beta_reps)}
    rows = [[0] * len(valid_reps) for _ in beta_reps]
    for j, aj in enumerate(valid_reps):
        acc = {}
        for bp, K in kraw_row(aj, r).items():
            rb = rep_of[bp]
            acc[rb] = acc.get(rb, 0) + K
        num0 = multinom(aj) * size[aj]
        for b, tot in acc.items():
            val, rem = divmod(num0 * tot, multinom(b) * size[b])
            if rem:
                raise RuntimeError(
                    f"reciprocity row build gave a non-integer at "
                    f"(alpha={aj}, beta={b}); the table is inconsistent")
            rows[bidx[b]][j] = val
    kinds = [("full", b) for b in beta_reps]

    # -- partial rows: every proper nonempty row subset S, every beta.
    #    K^S vanishes unless the S^c-marginals agree, so bucket the valid
    #    configurations by marginal and only touch the matching ones.
    if r >= 2:
        members = {a: orbit_members(a, r) for a in valid_reps}
        seen_rows = set()
        for S in range(1, m - 1):
            cs, cc, s_dim = _sub_index(S, r)
            ncos, nsub = 1 << (r - s_dim), 1 << s_dim
            # decompose each valid configuration into its cosets ONCE, and
            # bucket by S^c-marginal: K^S vanishes off the matching bucket.
            bucket = {}
            for j, a in enumerate(valid_reps):
                for aa in members[a]:
                    dec = _coset_split(aa, cs, cc, ncos, nsub, m)
                    bucket.setdefault(tuple(sum(g) for g in dec),
                                      []).append((j, tuple(map(tuple, dec))))
            for b in cfgs:
                dec_b = _coset_split(b, cs, cc, ncos, nsub, m)
                hit = bucket.get(tuple(sum(g) for g in dec_b))
                if not hit:
                    continue
                if s_dim == 1:
                    tabs = None
                    par = [(dec_b[w][1], dec_b[w][0] + dec_b[w][1])
                           for w in range(ncos)]
                else:
                    tabs = [_sub_kraw(tuple(dec_b[w]), s_dim)
                            for w in range(ncos)]
                row = [0] * len(valid_reps)
                for j, dec_a in hit:
                    k = 1
                    if tabs is None:
                        for w in range(ncos):
                            b1, tot = par[w]
                            k *= kraw1(dec_a[w][1], b1, tot)
                            if not k:
                                break
                    else:
                        for w in range(ncos):
                            k *= tabs[w].get(dec_a[w], 0)
                            if not k:
                                break
                    if k:
                        row[j] += k
                # the ORBIT sum can cancel to zero even when individual
                # members do not: test the assembled row, not the terms.
                if not any(row):
                    continue
                g = 0
                for v in row:
                    g = math.gcd(g, abs(v))
                key = tuple(v // g for v in row)
                if key in seen_rows:
                    continue
                seen_rows.add(key)
                rows.append(row)
                kinds.append((f"S{S}", b))

    # -- objectives
    c_obj = [0] * len(valid_reps)
    c_obj[0] = 1
    for k in range(1, n + 1):
        a = rep_of[tuple([n - k, k] + [0] * (m - 2))]
        j = col.get(a)
        if j is not None:
            c_obj[j] += math.comb(n, k)
    c_objp = [mass[j] for j in range(len(valid_reps))]
    n_full = len(beta_reps)

    if verbose:
        print(f"  DelsarteLin({r},{n},{d}) even={even}: "
              f"{len(valid_reps)} valid orbits, {len(rows)} constraint rows "
              f"({n_full} full + {len(rows) - n_full} partial)")
    return HierarchyLP(n, d, r, even, valid_reps, mass, rows, kinds,
                       c_obj, c_objp)


def solve_lp(lp: HierarchyLP, objective: str = "Obj"):
    """Solve the hierarchy LP in float64, returning a DUAL-consistent tuple.

    Returns ``(dual_value, phi, dual_multipliers)``.  Two HiGHS solves are
    used deliberately:

    * the primal solve supplies ``phi`` for support/contact diagnostics;
    * the dual is solved EXPLICITLY and supplies both the reported value and
      the multipliers that are later polished/certified.

    This avoids relying on marginals of a differently scaled primal model.
    Importantly, the primal variables have only the constraints present in
    LL Definition 5: phi_0 = 1 and phi_O >= 0.  There is NO phi_O <= 1
    constraint.  Adding that upper bound changes the LP and creates dual
    variables that the exact certifier does not contain.
    """
    import numpy as np
    from scipy.optimize import linprog

    if objective not in ("Obj", "Obj'"):
        raise ValueError(f"unknown objective {objective!r}")
    c = lp.c_obj if objective == "Obj" else lp.c_objp
    nv, nr = len(lp.valid_orbits), len(lp.rows)
    B = np.asarray(lp.rows, dtype=float)
    cf = np.asarray(c, dtype=float)

    # Primal: maximize c^T phi, B phi >= 0, phi_0 = 1, phi_j >= 0.
    row_scale = np.maximum(np.abs(B).max(axis=1), 1.0)
    pres = linprog(-cf, A_ub=-(B / row_scale[:, None]),
                   b_ub=np.zeros(nr),
                   bounds=[(1.0, 1.0)] + [(0.0, None)] * (nv - 1),
                   method="highs")
    if not pres.success:
        # The primal is diagnostic only.  Do not block a certifiable dual if
        # HiGHS has difficulty with the badly scaled primal at larger n.
        phi = [float("nan")] * nv
    else:
        phi = list(pres.x)

    # Explicit dual after eliminating phi_0 = 1:
    #   min c_0 + sum_i B[i,0] y_i
    #   s.t. sum_i B[i,j] y_i <= -c_j,  j >= 1,  y_i >= 0.
    # Scale each dual constraint only for the float solve.  Returned y is in
    # the coordinates of the original INTEGER rows.
    Adual = B[:, 1:].T
    bdual = -cf[1:]
    obj = B[:, 0]
    col_scale = np.maximum(np.abs(Adual).max(axis=1),
                           np.maximum(np.abs(bdual), 1.0))
    dres = linprog(obj, A_ub=Adual / col_scale[:, None],
                   b_ub=bdual / col_scale,
                   bounds=[(0.0, None)] * nr, method="highs")
    if not dres.success:
        raise RuntimeError(f"hierarchy dual LP failed: {dres.message}")
    value = float(c[0]) + float(dres.fun)
    return value, phi, list(dres.x)


@dataclass(frozen=True)
class DualPolishResult:
    """High-precision reconstruction of the HiGHS dual basic solution."""

    multipliers: tuple
    objective_value: object
    dps: int
    support: tuple[int, ...]
    tight_columns: tuple[int, ...]
    max_scaled_residual: object
    method: str = "active-set-mpmath"

    def report(self) -> str:
        k = len(self.tight_columns)
        extra = len(self.support) - k
        deg = f", {extra} fixed degenerate coord" + ("s" if extra != 1 else "") \
            if extra else ""
        return (f"{self.method}: {k}x{k} basis from support {len(self.support)}"
                f"{deg}, dps={self.dps}, max scaled violation "
                f"{float(self.max_scaled_residual):.3e}")


def polish_dual(lp: HierarchyLP, duals, objective: str = "Obj",
                dps: int = 100, support_tol: float = 1e-11,
                tight_tol: float = 1e-9, verbose: bool = True,
                rank_tol_rel: float = 1e-12) -> DualPolishResult:
    """Thin wrapper that guarantees the global mpmath precision is restored."""
    import mpmath as mp
    saved = mp.mp.dps
    try:
        return _polish_dual_impl(lp, duals, objective, dps, support_tol,
                                 tight_tol, verbose, rank_tol_rel)
    finally:
        mp.mp.dps = saved


def _polish_dual_impl(lp: HierarchyLP, duals, objective: str = "Obj",
                      dps: int = 100, support_tol: float = 1e-11,
                      tight_tol: float = 1e-9, verbose: bool = True,
                      rank_tol_rel: float = 1e-12) -> DualPolishResult:
    """Reconstruct a sparse dual basic solution at arbitrary precision.

    HiGHS supplies the *combinatorial* information: which multiplier rows are
    used and which dual column inequalities are tight.  We then discard any
    linear dependencies in that active set with rank-revealing QR and solve
    the resulting INTEGER square system with mpmath.  Degenerate HiGHS
    solutions are therefore fine: a 19-variable support of rank 18 is reduced
    to an 18x18 basic system rather than being treated as singular.

    No feasibility claim is made here.  ``certify`` still dyadically rounds,
    checks every inequality exactly, and applies the total repair if needed.
    A mistaken active set can therefore only worsen the bound, never validate
    a false one.
    """
    import mpmath as mp
    import numpy as np
    from scipy.linalg import qr

    if objective not in ("Obj", "Obj'"):
        raise ValueError(f"unknown objective {objective!r}")
    c = lp.c_obj if objective == "Obj" else lp.c_objp
    B = np.asarray(lp.rows, dtype=float)
    y0 = np.asarray([float(v) for v in duals], dtype=float)
    if len(y0) != len(lp.rows):
        raise ValueError("dual multiplier length does not match LP rows")

    ymax = max(1.0, float(np.max(np.abs(y0))) if len(y0) else 1.0)
    support0 = np.flatnonzero(y0 > support_tol * ymax)
    if len(support0) == 0:
        raise RuntimeError("cannot polish an empty dual support")

    # Dual column slacks: B[:,j]^T y + c_j <= 0, j >= 1.
    lhs = B[:, 1:].T @ y0 + np.asarray(c[1:], dtype=float)
    scale = np.maximum(np.max(np.abs(B[:, 1:]), axis=0),
                       np.maximum(np.abs(np.asarray(c[1:], dtype=float)), 1.0))
    scaled = lhs / scale
    tight0 = np.flatnonzero(np.abs(scaled) <= tight_tol)
    if len(tight0) == 0:
        raise RuntimeError("float dual has no identifiable tight columns")
    # `tight0` indexes `lhs`, whose entry k is LP COLUMN k + 1.  Every matrix
    # built from the active set has to use those columns -- `chosen` below
    # already shifts, so M0/M1 must shift too or the rank/pivot decisions are
    # taken on a different system from the one that is actually solved.
    tightcol = tight0 + 1

    # Active matrix: rows=tight dual constraints, cols=positive y variables.
    M0 = B[np.ix_(support0, tightcol)].T
    # First choose an independent subset of multiplier variables (columns).
    _, Rcol, pcol = qr(M0, mode="economic", pivoting=True)
    diag = np.abs(np.diag(Rcol))
    if len(diag) == 0:
        raise RuntimeError("empty active-set factorisation")
    rank_tol = rank_tol_rel * max(1.0, float(diag[0]))
    rank = int(np.sum(diag > rank_tol))
    if rank == 0:
        raise RuntimeError("active-set system has numerical rank zero")
    solve_support = np.asarray([support0[p] for p in pcol[:rank]], dtype=int)
    solve_set = set(int(i) for i in solve_support)
    fixed_support = np.asarray([int(i) for i in support0 if int(i) not in solve_set],
                               dtype=int)

    # Then choose rank independent tight equations (rows) for the variables we
    # solve.  Degenerate positive multipliers not needed for rank are kept at
    # their float values (lifted to high precision) and moved to the RHS.
    # This selects the member of the optimal affine face nearest the HiGHS
    # solution instead of arbitrarily forcing the redundant variables to 0.
    M1 = B[np.ix_(solve_support, tightcol)].T
    _, Rrow, prow = qr(M1.T, mode="economic", pivoting=True)
    diag2 = np.abs(np.diag(Rrow))
    if len(diag2) < rank or diag2[rank - 1] <= rank_tol_rel * max(1.0, diag2[0]):
        raise RuntimeError("could not extract a nonsingular active-set basis")
    chosen0 = np.asarray([tight0[p] for p in prow[:rank]], dtype=int)
    chosen = tuple(int(j) + 1 for j in chosen0)  # LP column ids incl. phi_0

    mp.mp.dps = int(dps)
    polished = [mp.mpf('0') for _ in lp.rows]
    for i in fixed_support:
        # repr(float) round-trips the exact binary64 value; this is only a
        # coordinate choice on a degenerate optimal face, not a feasibility
        # claim.  The exact certifier will still validate/repair everything.
        polished[int(i)] = mp.mpf(repr(float(y0[int(i)])))

    A_mp = mp.matrix(rank, rank)
    b_mp = mp.matrix(rank, 1)
    for rr, j in enumerate(chosen):
        rhs = mp.mpf(-int(c[j]))
        for i in fixed_support:
            rhs -= polished[int(i)] * int(lp.rows[int(i)][j])
        b_mp[rr] = rhs
        for cc, i in enumerate(solve_support):
            A_mp[rr, cc] = mp.mpf(int(lp.rows[int(i)][j]))
    sol = mp.lu_solve(A_mp, b_mp)

    neg_tol = mp.mpf(10) ** (-(max(20, dps // 2)))
    for cc, i in enumerate(solve_support):
        v = sol[cc]
        if v < -neg_tol:
            raise RuntimeError(
                f"polished basis has negative multiplier y[{int(i)}]="
                f"{mp.nstr(v, 12)}; try a different float basis")
        polished[int(i)] = max(mp.mpf('0'), v)
    support = support0

    # Measure only FEASIBILITY violation, not ordinary negative slack.
    max_violation = mp.mpf('0')
    for j in range(1, len(lp.valid_orbits)):
        rj = mp.mpf(int(c[j]))
        sj = mp.mpf('1')
        for i in support:
            aij = int(lp.rows[int(i)][j])
            rj += polished[int(i)] * aij
            sj = max(sj, mp.mpf(abs(aij)))
        if rj > 0:
            max_violation = max(max_violation, rj / sj)

    obj = mp.mpf(int(c[0]))
    for i in support:
        obj += polished[int(i)] * int(lp.rows[int(i)][0])
    result = DualPolishResult(tuple(polished), obj, int(dps),
                              tuple(int(i) for i in support), chosen,
                              max_violation)
    if verbose:
        print(f"  dual polish: {result.report()}, value "
              f"{mp.nstr(obj, 18)}")
    return result


# ---------------------------------------------------------------------------
# Exact certification
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HierarchyCertificate:
    n: int
    d: int
    r: int
    even: bool
    objective: str
    value: Fraction              # exact certified LP upper bound
    mu: Fraction                 # repair magnitude
    n_multipliers: int
    multipliers: tuple[Fraction, ...] = field(default_factory=tuple, repr=False)
    dual_values: tuple[Fraction, ...] = field(default_factory=tuple, repr=False)
    slacks: tuple[Fraction, ...] = field(default_factory=tuple, repr=False)

    def bound_on_A_lin(self) -> int:
        """floor of the certified value in the right power for A^Lin."""
        v = self.value
        if self.objective == "Obj":
            b = v.numerator // v.denominator
        else:                    # value bounds A^Lin(n,d)^r
            b = int(math.floor(float(v) ** (1.0 / self.r)))
            while Fraction((b + 1) ** self.r) <= v:
                b += 1
            while Fraction(b ** self.r) > v:
                b -= 1
        return int(b)

    def power_of_two_bound(self) -> int:
        """A^Lin is a power of 2, so the bound sharpens to 2^floor(lg)."""
        b = self.bound_on_A_lin()
        return int(1 << (b.bit_length() - 1)) if b >= 1 else 0

    def report(self) -> str:
        return (f"DelsarteLin({self.r},{self.n},{self.d}) [{self.objective}"
                f"{', even' if self.even else ''}]: certified value "
                f"{float(self.value):.6f}, so A^Lin({self.n},{self.d}) <= "
                f"{self.bound_on_A_lin()}, i.e. <= {self.power_of_two_bound()}")


def _dyadic_round(value, bits: int) -> Fraction:
    """Round float/Fraction/mpmath values to a dyadic WITHOUT float collapse."""
    unit = 1 << int(bits)
    if isinstance(value, Fraction):
        return Fraction(round(value * unit), unit)
    if isinstance(value, int):
        return Fraction(value, 1)
    # mpmath mpf values are EXACTLY dyadic, so convert them exactly instead of
    # through mpmath arithmetic.  Going via mp.nint(value * unit) silently caps
    # the resolution at the ambient global mp.mp.dps, which makes `bits` a
    # no-op beyond about 3.32 * dps and couples this function to whatever
    # precision some unrelated caller happened to leave behind.
    if hasattr(value, "_mpf_"):
        from mpmath.libmp import to_rational
        num, den = to_rational(value._mpf_)
        return Fraction(round(Fraction(num, den) * unit), unit)
    return Fraction(round(float(value) * unit), unit)


def certify(lp: HierarchyLP, duals, objective: str = "Obj",
            bits: int = 40, verbose: bool = True) -> HierarchyCertificate:
    """Round the float duals, repair exactly, return an exact certificate.

    Dual feasibility (derivation in the module docstring): y >= 0 and
    (B^T y)_O <= -c_O for every valid orbit O != O_0; certified value
    c_0 + (B^T y)_{O_0}.  Repair adds mu * y_hat with y_hat(beta) =
    multinom(beta) on the full rows, using the exact identity

        sum_beta multinom(beta) K_alpha(beta) = 2^{rn} [alpha = n eps_0],

    which is FIRST verified against the computed table (a failure means the
    table is wrong, and no certificate is produced).
    """
    n, r = lp.n, lp.r
    c = lp.c_obj if objective == "Obj" else lp.c_objp
    nv = len(lp.valid_orbits)
    # 1. verify the repair identity on the actual table (global table check)
    full_idx = [i for i, k in enumerate(lp.row_kind) if k[0] == "full"]
    w_beta = {}
    for i in full_idx:
        b = lp.row_kind[i][1]
        w_beta[i] = sum(multinom(bb) for bb in orbit_members(b, r))
    for j in range(nv):
        s = sum(w_beta[i] * lp.rows[i][j] for i in full_idx)
        # column j accumulates ALL alpha in its orbit, so the identity reads
        # sum_beta multinom(beta) * row_j(beta) = 2^{rn} [j = 0]: for j != 0
        # every member contributes 0, and the orbit sum of zeros is zero.
        want = (1 << (r * n)) if j == 0 else 0
        if s != want:
            raise RuntimeError(
                f"multinomial-sum identity FAILS at orbit {j}: {s} != {want}."
                f" The Krawtchouk table is inconsistent; refusing to certify.")

    # 2. rounded non-negative multipliers on the integer rows.  This path
    # preserves arbitrary-precision mpmath values from ``polish_dual``.
    y = [max(Fraction(0), _dyadic_round(v, bits)) for v in duals]

    # 3. exact (B^T y) and violation scan
    G = [sum(y[i] * lp.rows[i][j] for i in range(len(lp.rows)))
         for j in range(nv)]
    mu = Fraction(0)
    for j in range(1, nv):
        need = G[j] + c[j]           # must be <= 0; repair rate is -mass[j]
        if need > 0:
            mu = max(mu, Fraction(need, lp.orbit_mass[j]))

    if mu > 0:
        # The strictly feasible direction is the multinomial-weighted sum of
        # FULL rows with beta = n*eps_0 EXCLUDED.  Including that row would
        # make the nontrivial-column sum zero and therefore would NOT repair
        # anything.  Since the excluded zero-beta row equals orbit_mass, the
        # direction changes column 0 by 2^(rn)-1 and every j>0 by -mass[j].
        zero_beta = tuple([n] + [0] * ((1 << r) - 1))
        zero_full = [i for i in full_idx if lp.row_kind[i][1] == zero_beta]
        if len(zero_full) != 1:
            raise RuntimeError("could not identify the unique beta=n*eps_0 full row")
        repair_idx = [i for i in full_idx if i != zero_full[0]]
        for i in repair_idx:
            y[i] += mu * w_beta[i]

    # Recompute from the ACTUAL repaired multiplier vector.  Do not verify a
    # symbolic surrogate of the intended direction: the emitted y itself must
    # satisfy the inequalities.
    G = [sum(y[i] * lp.rows[i][j] for i in range(len(lp.rows)))
         for j in range(nv)]
    for j in range(1, nv):
        if G[j] + c[j] > 0:
            raise RuntimeError(
                f"repair failed at orbit {j}: G+c = {G[j] + c[j]}")

    value = Fraction(c[0]) + G[0]
    slacks = [Fraction(0)]
    slacks.extend(-(G[j] + c[j]) for j in range(1, nv))
    if verbose:
        print(f"  certified {objective} value {float(value):.6f} "
              f"(mu = {float(mu):.3e}, {len(y)} multipliers)")
    return HierarchyCertificate(n=lp.n, d=lp.d, r=r, even=lp.even,
                                objective=objective, value=value, mu=mu,
                                n_multipliers=len(y), multipliers=tuple(y),
                                dual_values=tuple(G), slacks=tuple(slacks))


def polish_and_certify(lp: HierarchyLP, duals, objective: str = "Obj",
                       dps: int = 100, bits: int = 180,
                       tight_tols=(1e-9,), rank_tols=(1e-12,),
                       verbose: bool = True):
    """Try several active-set readings and keep the one that certifies best.

    Which columns HiGHS reports as tight, and where the rank cut falls in the
    pivoted QR, are threshold decisions on a degenerate optimal face.  A bad
    reading does not endanger soundness -- certify() re-checks every
    inequality exactly and repairs -- it just inflates mu and therefore the
    bound.  Since certify() is cheap and its output is an exactly comparable
    Fraction, the selection criterion is simply the certified value.

    DEFAULT IS ONE READING.  Widen `tight_tols` / `rank_tols` only when a
    particular instance looks bad (a large mu, or a certified value visibly
    above the float LP value); the sweep costs a polish+certify per pair.

    Returns (best_polish, best_certificate).
    """
    best = None
    failures = []
    for tt in tight_tols:
        for rt in rank_tols:
            try:
                pol = polish_dual(lp, duals, objective, dps=dps,
                                  tight_tol=tt, rank_tol_rel=rt,
                                  verbose=False)
                cert = certify(lp, pol.multipliers, objective, bits=bits,
                               verbose=False)
            except (RuntimeError, ValueError) as exc:
                failures.append((tt, rt, str(exc)[:60]))
                continue
            if best is None or cert.value < best[1].value:
                best = (pol, cert, tt, rt)
    if best is None:
        raise RuntimeError(
            "every active-set reading failed: "
            + "; ".join(f"[{t:g}/{r:g}] {m}" for t, r, m in failures))
    pol, cert, tt, rt = best
    if verbose:
        print(f"  polish: {pol.report()}")
        print(f"  best of {len(tight_tols) * len(rank_tols)} readings "
              f"(tight_tol={tt:g}, rank_tol={rt:g}), "
              f"{len(failures)} failed")
    return pol, cert


# ---------------------------------------------------------------------------
# Dataset extraction for structural analysis / discovery
# ---------------------------------------------------------------------------
def _rate_log_abs(x: float, n: int):
    if not math.isfinite(x) or x == 0.0:
        return None
    return math.log(abs(x)) / n


def dual_dataset(lp: HierarchyLP, duals, phi,
                 certificate: HierarchyCertificate | None = None,
                 dual_threshold: float = 1e-14,
                 primal_threshold: float = 1e-10,
                 objective: str = "Obj") -> list[dict]:
    """Export multiplier, transformed-dual and contact-set information.

    Three object types are emitted:

    ``dual``
        y(beta), the multiplier measure on transform rows.
    ``dual_function``
        G(alpha) = sum_beta y_beta K_alpha(beta), together with the exact
        feasibility slack ``-G(alpha)-c(alpha)``.  This is invariant under
        non-uniqueness of a sparse multiplier representation and is often a
        better target for structural discovery.
    ``primal``
        the numerical phi support, retained only as complementary-slackness
        / contact-set context.

    If an exact ``certificate`` is supplied, y, G and slack come from its
    repaired rational vector rather than from the float solver.
    """
    n = lp.n
    obj = certificate.objective if certificate is not None else objective
    c = lp.c_obj if obj == "Obj" else lp.c_objp
    out = []

    if certificate is not None and certificate.multipliers:
        yvals = certificate.multipliers
        Gvals = certificate.dual_values
        svals = certificate.slacks
        exact = True
    else:
        yvals = tuple(duals)
        Gvals = []
        for j in range(len(lp.valid_orbits)):
            Gvals.append(sum(float(duals[i]) * lp.rows[i][j]
                             for i in range(len(lp.rows))))
        svals = [0.0] + [-(Gvals[j] + c[j])
                         for j in range(1, len(lp.valid_orbits))]
        exact = False

    for i, (kind, b) in enumerate(lp.row_kind):
        yf = float(yvals[i])
        if yf > dual_threshold:
            rec = {"role": "dual", "family": kind, "row_index": i,
                   "beta": list(b), "p": [bi / n for bi in b],
                   "weights": list(weights(b, lp.r)),
                   "weights_normalized": [w / n for w in weights(b, lp.r)],
                   "multiplier": yf,
                   "log_abs_multiplier_per_n": _rate_log_abs(yf, n)}
            if exact:
                q = yvals[i]
                rec["multiplier_exact"] = [int(q.numerator), int(q.denominator)]
            out.append(rec)

    for j, a in enumerate(lp.valid_orbits):
        gf, sf = float(Gvals[j]), float(svals[j])
        rec = {"role": "dual_function", "family": "G",
               "orbit_index": j, "alpha": list(a),
               "p": [ai / n for ai in a],
               "weights": list(weights(a, lp.r)),
               "weights_normalized": [w / n for w in weights(a, lp.r)],
               "orbit_mass": lp.orbit_mass[j], "objective_coefficient": c[j],
               "G": gf, "slack": sf,
               "log_abs_G_per_n": _rate_log_abs(gf, n),
               "contact": (svals[j] == 0 if exact else abs(sf) <= 1e-9)}
        if exact:
            qg, qs = Gvals[j], svals[j]
            rec["G_exact"] = [int(qg.numerator), int(qg.denominator)]
            rec["slack_exact"] = [int(qs.numerator), int(qs.denominator)]
        out.append(rec)

    if phi is not None:
        for j, a in enumerate(lp.valid_orbits):
            try:
                pf = float(phi[j])
            except (TypeError, ValueError):
                continue
            if math.isfinite(pf) and pf > primal_threshold:
                out.append({"role": "primal", "family": "phi",
                            "orbit_index": j, "alpha": list(a),
                            "p": [ai / n for ai in a],
                            "weights": list(weights(a, lp.r)),
                            "weights_normalized": [w / n for w in weights(a, lp.r)],
                            "orbit_mass": lp.orbit_mass[j],
                            "multiplier": pf})
    return out
