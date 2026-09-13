# Writing a problem plugin

The smallest useful problem specification implements three methods:

```python
from neuralcert import Evaluation, FunctionalProblem, Grid, SamplingConfig

class MyProblem(FunctionalProblem):
    name = "my-problem"

    def sample_domain(self, config: SamplingConfig) -> Grid:
        ...

    def evaluate(self, candidate, grid: Grid) -> Evaluation:
        # Return one differentiable scalar objective.
        ...

    def validate(self, candidate, grid: Grid):
        # Inexpensive checks, not a formal proof.
        return {"finite": True}
```

`evaluate` is the only place where problem-specific discovery mathematics is
required. The generic trainer manages models, the optimizer, device, dtype,
history, and best-state selection.

## Optional hooks

Problems may expose hooks for components that require specific mathematical
knowledge:

- `constraints(candidate, grid)` returns a differentiable penalty;
- `diagnostics(candidate, grid)` returns inexpensive numerical observations;
- `distill_rational(...)`, `distill_spectral(...)`, or `distill_sparse(...)`;
- `refine_eigen(...)`, `refine_convex(...)`, or `refine_newton(...)`;
- their own independent verifier callables.

These hooks are optional. NeuralCert makes no generic rationalization or
exactness claim when the problem does not provide enough information to support
one.

## Registration

```python
from neuralcert.core.registry import problems

problems.register("my-problem", MyProblem)
problem = problems.create("my-problem", ...)
```

Registration is convenient for CLIs and configuration files, but direct
construction of a problem object remains supported.

## Trust boundary

`validate` and landscape probes are numerical diagnostics. Only a separate
verifier may produce `VerificationResult(verified=True)`. Inject that verifier
into the pipeline; discovery must not promote its own result to a proof.
