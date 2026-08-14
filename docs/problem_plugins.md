# Een probleemplugin schrijven

De kleinste bruikbare problem spec implementeert drie methoden:

```python
from neuracert import Evaluation, FunctionalProblem, Grid, SamplingConfig

class MyProblem(FunctionalProblem):
    name = "my-problem"

    def sample_domain(self, config: SamplingConfig) -> Grid:
        ...

    def evaluate(self, candidate, grid: Grid) -> Evaluation:
        # Geef één differentiable scalar objective terug.
        ...

    def validate(self, candidate, grid: Grid):
        # Goedkope controles; geen formeel bewijs.
        return {"finite": True}
```

`evaluate` is de enige plaats waar probleemspecifieke discoverywiskunde nodig
is. De generieke trainer beheert modellen, optimizer, device, dtype, history en
best-state-selectie.

## Optionele hooks

Problemen kunnen hooks aanbieden voor componenten die specifieke wiskundige
kennis vereisen:

- `constraints(candidate, grid)` retourneert een differentiable penalty;
- `diagnostics(candidate, grid)` retourneert goedkope numerieke observaties;
- `distill_rational(...)`, `distill_spectral(...)` of `distill_sparse(...)`;
- `refine_eigen(...)`, `refine_convex(...)`, `refine_newton(...)`;
- eigen onafhankelijke verifier-callables.

Deze hooks zijn optioneel. NeuraCert doet geen generieke rationalisatie of
exactheidsclaim wanneer het probleem daarvoor onvoldoende informatie geeft.

## Registratie

```python
from neuracert.core.registry import problems

problems.register("my-problem", MyProblem)
problem = problems.create("my-problem", ...)
```

Registratie is handig voor CLI's en configuratiebestanden, maar directe
constructie van een problem object blijft ondersteund.

## Vertrouwensgrens

`validate` en landscape probes zijn numerieke diagnostiek. Alleen een
afzonderlijke verifier mag `VerificationResult(verified=True)` produceren.
Injecteer die verifier in de pipeline; laat discovery niet zijn eigen resultaat
tot bewijs promoveren.

