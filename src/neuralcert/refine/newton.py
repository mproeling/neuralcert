"""Newton-refinement delegation policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from neuralcert.core.result import DistillationResult, RefinementResult
from neuralcert.core.registry import refiners


@dataclass
class NewtonRefiner:
    tolerance: float = 1e-12
    max_iterations: int = 50

    def run(self, problem: Any, result: DistillationResult[Any]) -> RefinementResult[Any]:
        hook = getattr(problem, "refine_newton", None)
        if hook is None:
            raise NotImplementedError(f"{type(problem).__name__} has no Newton refiner")
        value = hook(result, tolerance=self.tolerance, max_iterations=self.max_iterations)
        return value if isinstance(value, RefinementResult) else RefinementResult(value=value)


refiners.register("newton", NewtonRefiner)
