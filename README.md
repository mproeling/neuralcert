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
maynard-discover --help
maynard-discover --method poly --help
maynard-discover --method ratio --help
maynard-discover-gated --help
maynard-certify --help
maynard-certify --method poly --help
maynard-certify --method ratio --help
maynard-certify-epsilon --help
maynard-certify-flint --help
maynard-certify-crt --help
maynard-certify-crt-ball --help
maynard-certify-crt-scaled --help
maynard-verify --help
maynard-verify-direct --help
```

De gated discovery kan zoals het oorspronkelijke script via `torchrun` worden
gestart. Het verouderde `--force-export`-argument is verwijderd.

Wanneer `--export PAD` is opgegeven, wordt het NPZ-bestand altijd geschreven.
Een mislukte numerieke refinement gate geeft daarbij een duidelijke waarschuwing
in de uitvoer, maar blokkeert de expliciet gevraagde export niet.

`maynard-discover` ondersteunt twee discovery-families. `--method poly` is de
bestaande neurale pipeline en blijft de standaard voor achterwaartse
compatibiliteit. `--method ratio` gebruikt de clustered confluent-ratio basis:

```bash
maynard-discover --method ratio --k 201 --mu 2,2,1 --export ratio-k201.npz
```

Dezelfde indeling geldt voor de hoofd-certificerings-CLI. De bestaande
Karatsuba-code valt onder `poly`; de nieuwe Arb-certifier valt onder `ratio`:

```bash
maynard-discover --method ratio --k 201 --mu 1 --export ratio-k201.npz
maynard-certify --method ratio ratio-k201.npz --cert-json ratio-k201.json
```

De ratio-certifier ondersteunt op dit moment uitsluitend een export met één
kanaal en macht 1. Gebruik daarom `--mu 1` bij discovery. Exports met meerdere
kanalen of hogere machten worden bewust geweigerd en niet stilzwijgend
vereenvoudigd.

## Onafhankelijke verifier

De verifier zit voor installatiegemak in dezelfde distributie, maar is een
zelfstandig subpackage. Hij importeert geen code uit discovery of certification
en implementeert de benodigde rekenstappen opnieuw.

```bash
maynard-verify poly-certificaat.json --method poly
maynard-verify ratio-certificaat.json --method ratio
```

Voor de niet-rigoureuze, directe Monte-Carlo-controle van de oorspronkelijke
Maynard-functionalen is er een bewust apart commando:

```bash
maynard-verify-direct --npz ratio-k201.npz --samples 1000000 --batches 40
```

`maynard-verify` controleert een bewijs; `maynard-verify-direct` is uitsluitend
een onafhankelijke diagnostische/falsificatietest en levert geen certificaat.

## Packagestructuur

```text
neuracert/
├── core/                 # problem/grid/result/constraint/registry-contracten
├── discovery/            # generieke modellen, trainer, optimizers en probes
├── distill/              # rational/spectral/sparse policies
├── refine/               # eigen/convex/Newton/local policies
├── verify/               # onafhankelijke verifierinterfaces en primitives
├── pipeline.py
└── problems/
    └── maynard/          # eerste probleemplugin en legacy adapters

maynard_tools/            # backwards-compatible gespecialiseerde implementatie
├── discovery/
├── certification/
└── verifier/
```

Zie [docs/architecture.md](docs/architecture.md) voor de dependencyregels.
