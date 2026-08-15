# Sign-uncertainty A_+(d): global LP results (session of 2026-08-13)

## The missing link
Every plateau (0.5785 Gaussian, 0.5929/0.5850 Laguerre) was a nonconvex
tau-optimizer basin. For the classical family, g(u) = e^u f(u) is a
POLYNOMIAL and the eigenspace constraint is LINEAR, so fixed-u0 feasibility
is convex: LP over Laguerre coefficients + bisection on u0 = global optimum
per degree, verified exactly by mpmath polyroots on the collocated function.

## Verified upper bounds (complete root lists, verdict PASS)
d=1 (published 0.572990, CG flag it unconverged):
  n=12 (deg 22): rho = 0.572989678   -- reproduces CG's configuration
  n=16 (deg 30): rho = 0.572706698   *** below published ***
  n=20 (deg 38): rho = 0.572588699   *** below published ***
       (10 LP tangencies; leave-one-out polish in rung 20, drop tau[6]=27.73)
d=2 (published 0.756207):
  n=12 (deg 22): rho = 0.756206236   *** below published (marginally) ***
       (6 LP tangencies; leave-one-out, drop tau[1]=12.00)

## Pipeline (laguerre_lp.py)
1. Margin-max LP oracle, exact-minimiser cutting planes, bisection on u0.
   Load-bearing details: NO column scaling (raw Laguerre coeffs are O(0.1);
   scaling inflated |c| to 3e7 and HiGHS silently violated rows by 7e-3);
   row equilibration to unit sup-norm; margin weights on the oscillatory
   window only; structural c_top >= 0 row (never far-anchor rows -- they are
   near-parallel to e_top and break the solver); loud refusal when a
   violation appears AT an existing constraint point.
2. Exact mpmath collocation at LP tangencies (square rung 2t+2), row-
   equilibrated, exact rational Laguerre coefficients.
3. verify_poly: ALL roots via polyroots + multiplicity clustering + midpoint
   signs => completeness statement no grid can give.
Extras: --polish taus (skip LP), BCK Thm 1 lower bound tripwire (0.4107675)
now in best_lower_bound, leave-one-out recipe when tangency count is off by
one (spurious shallow minimum near the feasibility boundary).

## Next commands
neuracert sign laguerre --d 1 --n-basis 24 --bisect 44 --dps 80 --json llp_d1_n24.json
neuracert sign laguerre --d 2 --n-basis 16 --bisect 44 --dps 70 --json llp_d2_n16.json
neuracert sign laguerre --d 2 --n-basis 20 --bisect 44 --dps 80 --json llp_d2_n20.json
# if tangency count != (n-2)/2: leave-one-out with
neuracert sign laguerre --d 1 --polish t1,...,tk --dps 80   # rung 2k+2
## Then
- certkit: Sturm/SOS over Q on the rationalised collocation -> theorem-grade.
- Hybrid A/B: add Gaussian columns to this LP; any gain below the polynomial
  family answers the hybrid hypothesis with no optimizer confound.


## Update 2: certification and the hybrid A/B (same session, continued)

### certkit stage complete -- the bounds are now THEOREMS over Q
certify_laguerre.py: taus rationalised to short denominators (the
certificate is exact for whatever rationals are chosen; full 52-bit dyadics
made the degree-38 Fraction elimination hang, 7-9 digits run in seconds),
exact Fraction collocation => P in Q[u] with exact double roots, exact
deflation P = prod (u-tau)^2 * R with zero remainders verified, then a
Sturm chain via primitive integer pseudo-remainder sequences (positive
multipliers only, so sign counts are untouched; naive Fraction remainders
square the bit-length per step and do not terminate in practice).

CERTIFIED (exact rational arithmetic end to end):
  A_+(1) <= sqrt(u_q/pi) = 0.572989678031   (deg 22)   cert_d1_n12.json
  A_+(1) <= 0.572706698600                  (deg 30)   cert_d1_n16.json
  A_+(1) <= 0.572588699103                  (deg 38)   cert_d1_n20.json
  A_+(2) <= 0.756206236766                  (deg 22)   cert_d2_n12.json
All four are exact rational certificates. At the precision of the published
Table 4.1 values, the degree-22 d=1 result reproduces the reported value;
the higher-degree d=1 results are genuine improvements.

### Hybrid A/B verdict (hybrid_lp.py): the polynomial family wins at d=1
Method: identical convex LP on both families (A = Laguerre block,
B = A + Gaussian widths), so the comparison is between families, not
optimisers.  Gaussian columns are first orthogonalised against the
Laguerre block on the working region (else the LP is numerically
rank-deficient: |c| ~ 6e8 of mutual cancellation, HiGHS residuals 2e-2).

The orthogonalisation IS the measurement.  Out-of-span component of
phi_a at n_basis = 12 (degree 22), d = 1:
  a = 1.5 : 1.3e-15   (machine zero -- contained)
  a = 2.5 : 3.0e-10   (contained)
  a = 4.0 : 9.2e-07   (contained for all practical purposes)
  a = 6.0 : 6.3e-05
  a = 10  : 1.6e-03
B-side global optimum with a = 6, 10: gain over pure = none beyond the
oracle noise floor; the genuinely-new directions enter with coefficients
~1e-3 and the constant does not move.

Interpretation: on the active region the Gaussian-mixture directions are
numerically CONTAINED in the polynomial family at fixed small d.  This
retroactively explains the entire Gaussian campaign: its 0.5785 plateau
was an awkward reparametrisation of (nearly) the same function space,
sitting slightly above the polynomial family's certified 0.5726.  The
mixture ansatz's natural home is the large-d asymptotic regime, exactly
where the CDG degree-o(d) obstruction bites the polynomials -- which is
the Krein/asymptotic question already queued in the roadmap.

### Files
laguerre_lp.py       global LP + collocation + polyroots verification
certify_laguerre.py  exact rational certification (Sturm over Z)
hybrid_lp.py         A/B test with orthogonalised mixed columns
cert_d1_n12/16/20.json, cert_d2_n12.json   the four certificates

## Reproducing each certified number from a single command

  d=1, deg 22 -> 0.572989678   neuracert sign laguerre --d 1 --n-basis 12 --bisect 42 --dps 60
  d=1, deg 30 -> 0.572706699   neuracert sign laguerre --d 1 --n-basis 16 --bisect 42 --dps 60
  d=1, deg 38 -> 0.572588699   neuracert sign laguerre --d 1 --n-basis 20 --bisect 42 --dps 70
  d=2, deg 22 -> 0.756206237   neuracert sign laguerre --d 2 --n-basis 12 --bisect 42 --dps 60

n_basis is the knob: rho falls with degree.  n=12 reproduces CG's own
configuration (5 contacts, degree 22) and only just dips below their value;
the improvement comes from n=16 and n=20.

The n=20 and d=2 runs need the leave-one-out repair, which is now automatic:
the margin LP reports one tangency too many (a shallow minimum sitting at the
margin level is indistinguishable from a true contact by depth alone).
Collocating all of them lands in the next rung, giving a valid but useless
function -- at n=20 a genuine PASS at rho = 9.34, last sign change dragged
out to u = 274.  The script now detects the disagreement with the LP estimate
and drops each candidate in turn, collocating the rest in the square rung.
At n=20, 4 of the 10 subsets tie at 0.5725886988 (the spurious contacts are
interchangeable -- itself the signature of the failure mode); the dropped one
is tau = 27.733562.

Certificates then follow from the printed taus:
  neuracert sign certify --d 1 --tau-digits 7 --taus <the 9 taus> \
      --out cert_d1_n20.json
