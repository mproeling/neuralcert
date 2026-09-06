# Cohn–Gonçalves sign uncertainty

The third bundled problem plugin studies the radial Fourier sign-uncertainty
constants `A_s(d)`. It contains separate numerical, structural and exact
stages rather than forcing these algorithms through the Maynard or Delsarte
command contracts.

## Workflows

Gaussian-mixture structure discovery:

```bash
neuralcert sign discover --mode single --d 4 --sign 1 --k 8 --json candidate.json
neuralcert sign collocate --d 4 --sign 1 --k 8 --json polished.json
```

Global Laguerre LP and exact certification:

```bash
neuralcert sign laguerre --d 1 --sign 1 --n-basis 12 --json candidate.json
neuralcert sign certify --json candidate.json --out certificate.json
neuralcert sign verify certificate.json
```

The Laguerre workflow automatically repairs one or two spurious shallow
contacts near a rung boundary by testing leave-out subsets in the square
collocation rung. Comparisons with published constants use the precision of
the reported decimal, so a sub-unit difference caused by rounding is labelled
as a reproduction rather than an improvement. See
[the Laguerre LP results appendix](sign-uncertainty-laguerre-results.md) for
the verified bounds, reproduction commands and numerical interpretation.

For higher dimensions and degrees, the LP margin is bounded so grid-only
recession directions can be found and cut instead of being misclassified as
an infeasible problem. The solver reports the finite-degree CDG lower bound as
a numerical tripwire and expands an infeasible upper bracket automatically.
Use `--u-lo` and `--u-hi` to override those brackets for reproduction runs.

The hybrid diagnostic compares the Laguerre space with added Gaussian
directions using the same convex oracle:

```bash
neuralcert sign hybrid --d 1 --n-basis 12 --widths 1.5,2.5,4,6,10
```

## Trust boundaries

`laguerre_basis.py` contains only the exact problem-level basis definition
shared by discovery and certification. The exact certifier does not import
the LP or Gaussian discovery modules. The verifier is more strongly isolated:
it imports only the Python standard library and independently reconstructs
the Laguerre polynomial, double contacts, exact deflation and Sturm chain.

At present theorem-grade certification supports `s=+1`, matching the supplied
certifier. Numerical discovery and the shared basis support both Fourier
signs. Gaussian-mixture and hybrid results remain numerical candidates unless
they are converted to an exact rational certificate.
