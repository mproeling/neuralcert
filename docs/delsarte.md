# Delsarte problem plugin

NeuraCert's second built-in problem is the Delsarte linear-programming bound
for codes in Hamming and Johnson association schemes. Its command line is
separate from Maynard because its inputs, numerical discovery and certificate
format are different.

```bash
neuracert-delsarte hamming --n 24 --distance 8 --q 2 \
  --certificate certificate.json
neuracert-verify-delsarte certificate.json
```

For constant-weight codes use the Johnson scheme:

```bash
neuracert-delsarte johnson --n 16 --distance 6 --weight 6 \
  --certificate certificate.json
```

The workflow solves a normalized floating-point LP, rounds its solution to
dyadic rationals, repairs feasibility exactly and emits a self-contained JSON
certificate. The independent verifier uses only Python's standard library and
imports no certificate-generation code from NeuraCert.

Python users can use `DelsarteCodeProblem.certify()` or the convenience
functions `binary_code_bound`, `qary_code_bound` and
`constant_weight_bound` from `neuracert.problems.delsarte`.

Shared association-scheme, CRT, basis, projection and exact-reporting code is
kept in `neuracert.exact`. This layer is available to future problem plugins;
the verifier deliberately remains outside that trust boundary.

## Higher-order binary hierarchy

The Loyfer-Linial/CJJ higher-order Delsarte LP is available separately:

```bash
neuracert-delsarte-hierarchy 10 6 --r 3 \
  --output hierarchy-r3-10-6.json
```

The v4 implementation computes genuine `GL(r,2)` orbits. In particular it
does not sort all seven nonzero coordinates at `r=3`, which would merge
distinct Fano-plane orbits. General partial transforms and the transposed
reciprocity-based table construction support `r=1` through `r=4`; practical
memory and LP size, rather than the orbit formula, are the limiting factors.

The hierarchy pipeline performs a floating-point solve, high-precision
active-set polishing, dyadic projection and exact dual repair. Supplying an
output path always writes both the certificate summary and the structural
dataset used for later discovery work.
