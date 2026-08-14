# NeuraCert

Een uitbreidbaar Python-framework voor numerieke discovery, distillation,
refinement en onafhankelijke verificatie van number-theoryproblemen.

Maynard is de eerste ingebouwde probleemplugin, niet langer de architectuur van
het hele package. De bestaande `maynard-*`-commando's blijven beschikbaar als
compatibiliteitslaag.

## Generieke discovery

Een gebruiker implementeert alleen sampling, een differentiable objective en
goedkope validatie:

```python
from neuracert import FunctionalProblem, Evaluation, Grid, discover

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

Voor een volledige workflow:

```python
from neuracert import Pipeline
from neuracert.discovery import NeuralDiscovery, LandscapeProbe
from neuracert.distill import RationalClusterDistiller
from neuracert.refine import GeneralizedEigenRefiner
from neuracert.verify import ExactCertificateVerifier

pipeline = Pipeline(
    discovery=NeuralDiscovery(...),
    diagnostics=LandscapeProbe(),
    distiller=RationalClusterDistiller(...),
    refiner=GeneralizedEigenRefiner(...),
    verifier=ExactCertificateVerifier(verifier=my_independent_verifier),
)
result = pipeline.run(MyProblem())
```

Zie [de pluginhandleiding](docs/problem_plugins.md) en
[het uitvoerbare voorbeeld](examples/custom_problem.py).

## Installatie

```bash
python -m pip install .
```

Voor de FLINT-backend:

```bash
python -m pip install ".[certification]"
```

Voor ontwikkeling en tests:

```bash
python -m pip install -e ".[dev,certification]"
python -m pytest
```

## Commando's

```bash
neuracert discover --help
neuracert discover --method poly --help
neuracert discover --method ratio --help
neuracert discover-gated --help
neuracert certify --help
neuracert certify --method poly --help
neuracert certify --method ratio --help
neuracert certify-epsilon --help
neuracert certify-flint --help
neuracert certify-crt --help
neuracert certify-crt-ball --help
neuracert certify-crt-scaled --help
neuracert verify --help
neuracert verify-direct --help
```

De gated discovery kan zoals het oorspronkelijke script via `torchrun` worden
gestart. Het verouderde `--force-export`-argument is verwijderd.

Wanneer `--export PAD` is opgegeven, wordt het NPZ-bestand altijd geschreven.
Een mislukte numerieke refinement gate geeft daarbij een duidelijke waarschuwing
in de uitvoer, maar blokkeert de expliciet gevraagde export niet.

`neuracert discover` ondersteunt twee discovery-families. `--method poly` is de
bestaande neurale pipeline en blijft de standaard voor achterwaartse
compatibiliteit. `--method ratio` gebruikt de clustered confluent-ratio basis:

```bash
neuracert discover --method ratio --k 201 --mu 2,2,1 --export ratio-k201.npz
```

Dezelfde indeling geldt voor de hoofd-certificerings-CLI. De bestaande
Karatsuba-code valt onder `poly`; de nieuwe Arb-certifier valt onder `ratio`:

```bash
neuracert discover --method ratio --k 201 --mu 1 --export ratio-k201.npz
neuracert certify --method ratio ratio-k201.npz --cert-json ratio-k201.json
```

De ratio-certifier accepteert discovery-exports ook expliciet via `--npz`:

```bash
neuracert certify --method ratio --npz k650000000_single.npz
```

Deze backend gebruikt de v6 Arb-route met directed rounding end-to-end. De
NPZ-inhoud en SHA-256 worden vóór certificering gecontroleerd.

De ratio-certifier ondersteunt op dit moment uitsluitend een export met één
kanaal en macht 1. Gebruik daarom `--mu 1` bij discovery. Exports met meerdere
kanalen of hogere machten worden bewust geweigerd en niet stilzwijgend
vereenvoudigd.

## Onafhankelijke verifier

De verifier zit voor installatiegemak in dezelfde distributie, maar is een
zelfstandig subpackage. Hij importeert geen code uit discovery of certification
en implementeert de benodigde rekenstappen opnieuw.

```bash
neuracert verify poly-certificaat.json --method poly
neuracert verify ratio-certificaat.json --method ratio
```

Voor de niet-rigoureuze, directe Monte-Carlo-controle van de oorspronkelijke
Maynard-functionalen is er een bewust apart commando:

```bash
neuracert verify-direct --npz ratio-k201.npz --samples 1000000 --batches 40
```

`neuracert verify` controleert een bewijs; `neuracert verify-direct` is uitsluitend
een onafhankelijke diagnostische/falsificatietest en levert geen certificaat.

### Delsarte-codegrenzen

Delsarte is als tweede, zelfstandige probleemplugin toegevoegd. De eigen CLI
weerspiegelt dat dit probleem andere invoer en een andere certificeringsroute
heeft dan Maynard:

```bash
neuracert delsarte bound hamming --n 24 --distance 8 --q 2 \
  --certificate delsarte.json
neuracert delsarte verify delsarte.json

neuracert delsarte bound johnson --n 16 --distance 6 --weight 6 \
  --certificate constant-weight.json

neuracert delsarte hierarchy 10 6 --r 3 \
  --output hierarchy-r3-10-6.json
```

De LP wordt numeriek opgelost, dyadisch afgerond en daarna exact gerepareerd.
De JSON-uitvoer is zelfvoorzienend; de Delsarte-verifier importeert geen
NeuraCert-code en gebruikt alleen de Python-standaardbibliotheek. Zie
[docs/delsarte.md](docs/delsarte.md) voor de Python-API en de trust boundary.

### Cohn–Gonçalves tekenonzekerheid

De derde probleemplugin bevat Gaussian-mixture discovery, collocatie, een
globale Laguerre-LP, een hybride A/B-diagnose en exacte Sturm-certificering:

```bash
neuracert sign laguerre --d 1 --sign 1 --n-basis 12 --json candidate.json
neuracert sign certify --json candidate.json --out certificate.json
neuracert sign verify certificate.json
```

Zie [docs/sign-uncertainty.md](docs/sign-uncertainty.md) voor alle discovery-
en diagnosecommando's en de scheiding tussen discovery, certificering en
onafhankelijke verificatie.

## Packagestructuur

```text
neuracert/
├── core/                 # problem/grid/result/constraint/registry-contracten
├── exact/                # herbruikbare CRT-, basis- en schemes-infrastructuur
├── discovery/            # generieke modellen, trainer, optimizers en probes
├── distill/              # rational/spectral/sparse policies
├── refine/               # eigen/convex/Newton/local policies
├── verify/               # onafhankelijke verifierinterfaces en primitives
├── pipeline.py
└── problems/
    ├── maynard/          # Maynard-plugin en legacy adapters
    ├── delsarte/         # niveau-1 en hogere LP, certificering en losse verifier
    └── sign_uncertainty/ # Gaussian/Laguerre discovery en exacte Sturm-verificatie

maynard_tools/            # backwards-compatible gespecialiseerde implementatie
├── discovery/
├── certification/
└── verifier/
```

Zie [docs/architecture.md](docs/architecture.md) voor de dependencyregels.
