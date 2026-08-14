# Maynard Tools

Een installeerbaar Python-package voor twee strikt gescheiden workflows:

- **discovery** zoekt met floating-point-optimalisatie naar geschikte
  separeerbare trial functions;
- **certification** controleert geëxporteerde kandidaten met exacte of
  intervalrekenkunde.

Discovery importeert nooit uit certification. Een discovery-resultaat is dus
geen impliciet certificaat en de exacte backends kunnen onafhankelijk worden
geïnstalleerd en uitgevoerd.

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
maynard_tools/
├── discovery/
│   ├── numerics.py       # quadratuur, interpolatie en chain planning
│   ├── cli.py            # dispatcher voor --method poly|ratio
│   ├── models.py         # gedeelde PyTorch-kanaalmodellen
│   ├── diagnostics.py    # conditionering en referentiecontroles
│   ├── distributed.py    # procesgroepen en pair-sharding
│   ├── scheduling.py     # iteratie- en learning-ratehelpers
│   ├── factored.py       # factored discovery-pipeline
│   ├── ratio.py          # clustered confluent-ratio discovery
│   └── gated.py          # gated/distributed discovery-pipeline
├── certification/
    ├── cli.py
    ├── ratio.py
    ├── karatsuba.py
    ├── epsilon_karatsuba.py
    ├── flint_streaming.py
    ├── crt.py
    ├── crt_ball.py
    ├── crt_ball_scaled.py
    ├── scaled_crt.py
    └── scaled_domain.py
└── verifier/
    ├── certificate.py    # rigoureuze poly/ratio-certificaatverificatie
    └── direct_ij.py      # niet-rigoureuze directe I/J-controle
```

Zie [docs/architecture.md](docs/architecture.md) voor de dependencyregels.
