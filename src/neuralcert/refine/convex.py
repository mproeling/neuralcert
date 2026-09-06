"""Convex-refinement delegation policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from neuralcert.core.result import DistillationResult, RefinementResult
from neuralcert.core.registry import refiners


@dataclass
class ConvexRefiner:
    solver: str | None = None

    def run(self, problem: Any, result: DistillationResult[Any]) -> RefinementResult[Any]:
        hook = getattr(problem, "refine_convex", None)
        if hook is None:
            raise NotImplementedError(f"{type(problem).__name__} has no convex refiner")
        value = hook(result, solver=self.solver)
        return value if isinstance(value, RefinementResult) else RefinementResult(value=value)


refiners.register("convex", ConvexRefiner)
