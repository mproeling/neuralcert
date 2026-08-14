# Cohn–Gonçalves sign uncertainty

The third bundled problem plugin studies the radial Fourier sign-uncertainty
constants `A_s(d)`. It contains separate numerical, structural and exact
stages rather than forcing these algorithms through the Maynard or Delsarte
command contracts.

## Workflows

Gaussian-mixture structure discovery:

```bash
neuracert sign discover --mode single --d 4 --sign 1 --k 8 --json candidate.json
neuracert sign collocate --d 4 --sign 1 --k 8 --json polished.json
```

Global Laguerre LP and exact certification:

```bash
neuracert sign laguerre --d 1 --sign 1 --n-basis 12 --json candidate.json
neuracert sign certify --json candidate.json --out certificate.json
neuracert sign verify certificate.json
```

The hybrid diagnostic compares the Laguerre space with added Gaussian
directions using the same convex oracle:

```bash
neuracert sign hybrid --d 1 --n-basis 12 --widths 1.5,2.5,4,6,10
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
