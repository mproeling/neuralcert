"""Generalized-eigenvalue refinement contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from neuracert.core.registry import refiners
from neuracert.core.result import DistillationResult, RefinementResult


@dataclass
class GeneralizedEigenRefiner:
    rank_tolerance: float = 1e-12
    stability_tolerance: float = 1e-8

    def run(self, problem: Any, result: DistillationResult[Any]) -> RefinementResult[Any]:
        hook = getattr(problem, "refine_eigen", None)
        if hook is None:
            raise NotImplementedError(
                f"{type(problem).__name__} must implement refine_eigen()"
            )
        value = hook(result, rank_tolerance=self.rank_tolerance,
                     stability_tolerance=self.stability_tolerance)
        return value if isinstance(value, RefinementResult) else RefinementResult(
            value=value, metadata={"refiner": "generalized-eigen"}
        )


refiners.register("eigen", GeneralizedEigenRefiner)

