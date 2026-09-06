# NeuralCert

Een uitbreidbaar Python-framework voor numerieke discovery, distillation,
refinement en onafhankelijke verificatie van number-theoryproblemen.

Maynard is de eerste ingebouwde probleemplugin, niet langer de architectuur van
het hele package. De bestaande `maynard-*`-commando's blijven beschikbaar als
compatibiliteitslaag.

## Generieke discovery

Een gebruiker implementeert alleen sampling, een differentiable objective en
goedkope validatie:

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

Voor een volledige workflow:

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

Zie [de pluginhandleiding](docs/problem_plugins.md) en
[het uitvoerbare voorbeeld](examples/custom_problem.py).

## Installatie

```bash
python -m pip install .
```

`python-flint` wordt standaard meegeïnstalleerd, omdat exacte certificering
`from flint import ...` gebruikt. De distributienaam voor `pip` is dus
`python-flint`, terwijl de Python-importnaam `flint` is.

Voor ontwikkeling en tests:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

### Migratie van `neuracert` naar `neuralcert`

Versie 0.11.0 wijzigt zowel de distributienaam, Python-namespace als CLI. Na
een upgrade verwijder je daarom eerst de oude installatie:

```bash
python -m pip uninstall neuracert
python -m pip install .
```

Vervang vervolgens `import neuracert` door `import neuralcert` en commando's
zoals `neuracert discover` door `neuralcert discover`. Er wordt bewust geen
oude import- of CLI-alias geïnstalleerd, zodat nieuwe omgevingen nog maar één
publieke naam bevatten.

## Commando's

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

De gated discovery kan zoals het oorspronkelijke script via `torchrun` worden
gestart. Het verouderde `--force-export`-argument is verwijderd.

Wanneer `--export PAD` is opgegeven, wordt het NPZ-bestand altijd geschreven.
Een mislukte numerieke refinement gate geeft daarbij een duidelijke waarschuwing
in de uitvoer, maar blokkeert de expliciet gevraagde export niet.

`neuralcert discover` ondersteunt twee methodologische families:

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
neuralcert discover --method ratio --k 201 --mu 2,2,1 --export ratio-k201.npz
```

De ratio-discovery ondersteunt ook de vergrote simplex met `--epsilon`:

```bash
neuralcert discover --method ratio --k 201 --mu 1 --epsilon 0.01 \
  --export ratio-epsilon.npz
```

Een epsilon-export legt epsilon expliciet in het NPZ-bestand en de hash vast.
De huidige v6 Arb-certifier ondersteunt alleen `epsilon=0` en weigert een
epsilon-export daarom expliciet; hij zal die nooit stilzwijgend als het gewone
Maynard-probleem certificeren.

Dezelfde indeling geldt voor de hoofd-certificerings-CLI. De bestaande
Karatsuba-code valt onder `poly`; de nieuwe Arb-certifier valt onder `ratio`:

```bash
neuralcert discover --method ratio --k 201 --mu 1 --export ratio-k201.npz
neuralcert certify --method ratio ratio-k201.npz --cert-json ratio-k201.json
```

De ratio-certifier accepteert discovery-exports ook expliciet via `--npz`:

```bash
neuralcert certify --method ratio --npz k650000000_single.npz
```

Bij `neuralcert certify --npz bestand.npz` wordt de methode automatisch uit het
NPZ-schema herkend. Exacte ratio-kanalen worden uit de gehashte `canonical`-
tekst gereconstrueerd; binaire objectvelden in oudere exports worden niet
geopend en nieuwe exports bevatten uitsluitend numerieke of Unicode-arrays.

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
neuralcert verify poly-certificaat.json --method poly
neuralcert verify ratio-certificaat.json --method ratio
```

Voor de niet-rigoureuze, directe Monte-Carlo-controle van de oorspronkelijke
Maynard-functionalen is er een bewust apart commando:

```bash
neuralcert verify-direct --npz ratio-k201.npz --samples 1000000 --batches 40
```

`neuralcert verify` controleert een bewijs; `neuralcert verify-direct` is uitsluitend
een onafhankelijke diagnostische/falsificatietest en levert geen certificaat.

### Delsarte-codegrenzen

Delsarte is als tweede, zelfstandige probleemplugin toegevoegd. De eigen CLI
weerspiegelt dat dit probleem andere invoer en een andere certificeringsroute
heeft dan Maynard:

```bash
neuralcert delsarte bound hamming --n 24 --distance 8 --q 2 \
  --certificate delsarte.json
neuralcert delsarte verify delsarte.json

neuralcert delsarte bound johnson --n 16 --distance 6 --weight 6 \
  --certificate constant-weight.json

neuralcert delsarte hierarchy 10 6 --r 3 \
  --output hierarchy-r3-10-6.json
```

De LP wordt numeriek opgelost, dyadisch afgerond en daarna exact gerepareerd.
De JSON-uitvoer is zelfvoorzienend; de Delsarte-verifier importeert geen
NeuralCert-code en gebruikt alleen de Python-standaardbibliotheek. Zie
[docs/delsarte.md](docs/delsarte.md) voor de Python-API en de trust boundary.

### Cohn–Gonçalves tekenonzekerheid

De derde probleemplugin bevat Gaussian-mixture discovery, collocatie, een
globale Laguerre-LP, een hybride A/B-diagnose en exacte Sturm-certificering:

```bash
neuralcert sign laguerre --d 1 --sign 1 --n-basis 12 --json candidate.json
neuralcert sign certify --json candidate.json --out certificate.json
neuralcert sign verify certificate.json
```

Zie [docs/sign-uncertainty.md](docs/sign-uncertainty.md) voor alle discovery-
en diagnosecommando's en de scheiding tussen discovery, certificering en
onafhankelijke verificatie.

### Meegeleverde datasets

De Maynard eta-sweeps worden als package-resources meegeleverd en zijn dus ook
beschikbaar na installatie uit een wheel:

```python
import csv
from neuralcert.data import dataset

with dataset("maynard_geometric_eta_sweep").open(
    "r", encoding="utf-8", newline=""
) as stream:
    rows = list(csv.DictReader(stream))
```

Beschikbare namen zijn `maynard_geometric_eta_sweep` en
`maynard_R_sweep_eta`. De exacte bestandsnaam met `.csv` wordt eveneens
geaccepteerd.

## Packagestructuur

```text
neuralcert/
├── core/                 # problem/grid/result/constraint/registry-contracten
├── exact/                # herbruikbare CRT-, basis- en schemes-infrastructuur
├── discovery/            # generieke modellen, trainer, optimizers en probes
├── distill/              # rational/spectral/sparse policies
├── refine/               # eigen/convex/Newton/local policies
├── verify/               # onafhankelijke verifierinterfaces en primitives
├── data/                 # meegeleverde reproduceerbare CSV-datasets
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
