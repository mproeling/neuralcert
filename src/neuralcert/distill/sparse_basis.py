"""Sparse-basis distillation policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from neuralcert.core.result import DiscoveryResult, DistillationResult
from neuralcert.core.registry import distillers


@dataclass
class SparseBasisDistiller:
    max_terms: int = 32
    tolerance: float = 1e-10

    def run(self, problem: Any, result: DiscoveryResult[Any]) -> DistillationResult[Any]:
        hook = getattr(problem, "distill_sparse", None)
        if hook is None:
            raise NotImplementedError(
                f"{type(problem).__name__} must implement distill_sparse()"
            )
        value = hook(result, max_terms=self.max_terms, tolerance=self.tolerance)
        return value if isinstance(value, DistillationResult) else DistillationResult(
            value=value, metadata={"distiller": "sparse"}
        )


distillers.register("sparse", SparseBasisDistiller)
