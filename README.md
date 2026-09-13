# NeuralCert

An extensible Python framework for numerical discovery, distillation,
refinement, and independent verification of number-theory problems.

Maynard is the first bundled problem plugin rather than the architecture of the
entire package. The existing `maynard-*` commands remain available as a
compatibility layer.

## Generic discovery

Users need only implement sampling, a differentiable objective, and inexpensive
validation:

```python
from neuralcert import FunctionalProblem, Evaluation, Grid, discover

class MyProblem(FunctionalProblem):
    name = "my-number-theory-problem"

    def sample_domain(self, config):
        return Grid(points=...)

    def evaluate(self, candidate, grid):
        return Evaluation(objective=...)

    def validate(self, candidate, grid):
        return {"finite": True}

result = discover(problem=MyProblem(), model="mlp", device="cuda")
```

For a complete workflow:

```python
from neuralcert import Pipeline
from neuralcert.discovery import NeuralDiscovery, LandscapeProbe
from neuralcert.distill import RationalClusterDistiller
from neuralcert.refine import GeneralizedEigenRefiner
from neuralcert.verify import ExactCertificateVerifier

pipeline = Pipeline(
    discovery=NeuralDiscovery(...),
    diagnostics=LandscapeProbe(),
    distiller=RationalClusterDistiller(...),
    refiner=GeneralizedEigenRefiner(...),
    verifier=ExactCertificateVerifier(verifier=my_independent_verifier),
)
result = pipeline.run(MyProblem())
```

See the [problem-plugin guide](docs/problem_plugins.md) and the
[executable example](examples/custom_problem.py).

## Installation

```bash
python -m pip install .
```

`python-flint` is installed by default because exact certification uses
`from flint import ...`. Its distribution name on `pip` is therefore
`python-flint`, while its Python import name is `flint`.

For development and testing:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

### Migrating from `neuracert` to `neuralcert`

Version 0.11.0 changed the distribution name, Python namespace, and CLI. Remove
the old installation before upgrading:

```bash
python -m pip uninstall neuracert
python -m pip install .
```

Then replace `import neuracert` with `import neuralcert`, and commands such as
`neuracert discover` with `neuralcert discover`. No legacy import or CLI alias
is installed, so new environments expose only one public name.

## Commands

```bash
neuralcert discover --help
neuralcert discover --method poly --help
neuralcert discover --method ratio --help
neuralcert discover-gated --help
neuralcert certify --help
neuralcert certify --method poly --help
neuralcert certify --method ratio --help
neuralcert certify-epsilon --help
neuralcert certify-flint --help
neuralcert certify-crt --help
neuralcert certify-crt-ball --help
neuralcert certify-crt-scaled --help
neuralcert verify --help
neuralcert verify-direct --help
```

As in the original script, gated discovery can be launched through `torchrun`.
The obsolete `--force-export` argument has been removed.

Whenever `--export PATH` is supplied, the NPZ file is written. A failed
numerical refinement gate produces a clear warning but does not block an
explicitly requested export.

`neuralcert discover` supports two methodological families:

`--method poly`
: Polynomial channels: symmetric-polynomial trial functions optimised by Adam
  followed by L-BFGS, certified by exact multimodular (CRT) evaluation of the
  Gram forms. Strongest at small `k`, where the extremal function is genuinely
  high-dimensional; cost grows with degree and channel count.

`--method ratio`
: Rational channels `g(t) = 1/(c + (k-1)t)`: a one-parameter family evaluated
  analytically via its characteristic function and a single FFT, certified in
  Arb ball arithmetic. Cost independent of `k`, so it reaches `k ~ 1e9`, and
  it attains `log k - 0.334 + o(1)` asymptotically.

```bash
neuralcert discover --method ratio --opt direct --k 201 --mu 2,2,1 \
  --export ratio-k201.npz
```

`--opt direct` is the default and preserves the original route: the rational
family is optimized directly against the Rayleigh quotient, with no neural
intermediate step.

A neural `poly` export can first be distilled into this clustered rational
family by weighted variable projection. For each fixed set of pole locations,
the linear coefficients are eliminated by least squares; only the positive,
ordered poles are optimized nonlinearly:

```bash
neuralcert discover --method ratio --opt neural --k 201 --mu 2,2,1 \
  --distill-npz neural-k201.npz --export ratio-k201.npz
```

For multiple neural channels, the channel with the largest absolute Ritz
coefficient is selected by default. `--distill-channel INDEX` explicitly
selects another channel. `--distill-poles`, `--distill-maxiter`, and
`--distill-prune` control the initial poles, outer optimization budget, and
multiplicity pruning, respectively. With `--opt neural`, the fit determines
only the initial structure; the reported `R` is subsequently reoptimized in
the deterministic ratio evaluator.

Ratio discovery also supports the enlarged simplex through `--epsilon`:

```bash
neuralcert discover --method ratio --k 201 --mu 1 --epsilon 0.01 \
  --export ratio-epsilon.npz
```

An epsilon export records epsilon explicitly in both the NPZ file and its hash.
The current v6 Arb certifier supports only `epsilon=0` and therefore rejects an
epsilon export explicitly; it will never silently certify it as the ordinary
Maynard problem.

The same division applies to the main certification CLI. The existing
Karatsuba implementation belongs to `poly`; the Arb certifier belongs to
`ratio`:

```bash
neuralcert discover --method ratio --k 201 --mu 1 --export ratio-k201.npz
neuralcert certify --method ratio ratio-k201.npz --cert-json ratio-k201.json
```

The ratio certifier also accepts discovery exports explicitly through `--npz`:

```bash
neuralcert certify --method ratio --npz k650000000_single.npz
```

With `neuralcert certify --npz FILE.npz`, the method is detected automatically
from the NPZ schema. Exact ratio channels are reconstructed from the hashed
`canonical` text; binary object fields in older exports are not opened, and
new exports contain only numeric or Unicode arrays.

This backend uses the v6 Arb route with directed rounding end to end. The NPZ
contents and SHA-256 digest are checked before certification.

The ratio certifier currently supports only a single-channel, power-1 export.
Use `--mu 1` during discovery. Exports with multiple channels or higher powers
are deliberately rejected rather than silently simplified.

For polynomial discovery exports, every certification backend reconstructs the
same channel frame through one shared loader. `channel_norms` normalizes only
`g_fine`; an exported Ritz vector is mapped back to that normalized frame with
`exp(-logA_diag/2)`. This also applies to `--use-nn-c`; the default re-Ritz
route remains available.

## Independent verifier

For installation convenience, the verifier is distributed in the same wheel,
but it is a standalone subpackage. It imports no discovery or certification
code and independently reimplements the required computations.

```bash
neuralcert verify poly-certificate.json --method poly
neuralcert verify ratio-certificate.json --method ratio
```

The non-rigorous direct Monte Carlo check of the original Maynard functionals
is intentionally exposed as a separate command:

```bash
neuralcert verify-direct --npz ratio-k201.npz --samples 1000000 --batches 40
```

`neuralcert verify` checks a proof, whereas `neuralcert verify-direct` is only
an independent diagnostic and falsification test; it does not produce a
certificate.

### Delsarte code bounds

Delsarte is the second standalone problem plugin. Its dedicated CLI reflects
that this problem has different inputs and a different certification route
from Maynard:

```bash
neuralcert delsarte bound hamming --n 24 --distance 8 --q 2 \
  --certificate delsarte.json
neuralcert delsarte verify delsarte.json

neuralcert delsarte bound johnson --n 16 --distance 6 --weight 6 \
  --certificate constant-weight.json

neuralcert delsarte hierarchy 10 6 --r 3 \
  --output hierarchy-r3-10-6.json
```

The LP is solved numerically, rounded to dyadic rationals, and then repaired
exactly. The JSON output is self-contained; the Delsarte verifier imports no
NeuralCert code and uses only the Python standard library. See
[docs/delsarte.md](docs/delsarte.md) for the Python API and trust boundary.

### Cohn–Gonçalves sign uncertainty

The third problem plugin contains Gaussian-mixture discovery, collocation, a
global Laguerre LP, a hybrid A/B diagnostic, and exact Sturm certification:

```bash
neuralcert sign laguerre --d 1 --sign 1 --n-basis 12 --json candidate.json
neuralcert sign certify --json candidate.json --out certificate.json
neuralcert sign verify certificate.json
```

See [docs/sign-uncertainty.md](docs/sign-uncertainty.md) for all discovery and
diagnostic commands and for the separation between discovery, certification,
and independent verification.

### Bundled datasets

The Maynard eta sweeps are bundled as package resources and therefore remain
available after installation from a wheel:

```python
import csv
from neuralcert.data import dataset

with dataset("maynard_geometric_eta_sweep").open(
    "r", encoding="utf-8", newline=""
) as stream:
    rows = list(csv.DictReader(stream))
```

Available names are `maynard_geometric_eta_sweep` and `maynard_R_sweep_eta`.
The exact filename including `.csv` is also accepted.

## Package structure

```text
neuralcert/
├── core/                 # problem/grid/result/constraint/registry contracts
├── exact/                # reusable CRT, basis, and scheme infrastructure
├── discovery/            # generic models, trainer, optimizers, and probes
├── distill/              # rational/spectral/sparse policies
├── refine/               # eigen/convex/Newton/local policies
├── verify/               # independent verifier interfaces and primitives
├── data/                 # bundled reproducible CSV datasets
├── pipeline.py
└── problems/
    ├── maynard/          # Maynard plugin and legacy adapters
    ├── delsarte/         # level-1/higher LP, certification, standalone verifier
    └── sign_uncertainty/ # Gaussian/Laguerre discovery and exact Sturm verification

maynard_tools/            # backward-compatible specialized implementation
├── discovery/
├── certification/
└── verifier/
```

See [docs/architecture.md](docs/architecture.md) for the dependency rules.
