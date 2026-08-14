"""Generic delegation for local refinement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from neuracert.core.result import DistillationResult, RefinementResult
from neuracert.core.registry import refiners


@dataclass
class LocalRefiner:
    method: str = "problem-default"
    max_iterations: int = 100

    def run(self, problem: Any, result: DistillationResult[Any]) -> RefinementResult[Any]:
        hook = getattr(problem, "refine_local", None) or getattr(problem, "refine", None)
        if hook is None:
            raise NotImplementedError(f"{type(problem).__name__} has no local refiner")
        try:
            value = hook(result, method=self.method, max_iterations=self.max_iterations)
        except TypeError:
            value = hook(result)
        return value if isinstance(value, RefinementResult) else RefinementResult(
            value=value, metadata={"refiner": "local", "method": self.method}
        )


refiners.register("local", LocalRefiner)
