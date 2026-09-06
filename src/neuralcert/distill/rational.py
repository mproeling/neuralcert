"""Configuration shell for rational-cluster distillation plugins."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from neuralcert.core.registry import distillers
from neuralcert.core.result import DiscoveryResult, DistillationResult


@dataclass
class RationalClusterDistiller:
    """Request a problem-specific rational representation.

    Rational approximation is domain-sensitive, so this generic component
    supplies policy while the problem owns the mathematical conversion.
    """

    max_denominator: int = 10**12
    tolerance: float = 1e-10
    prune_tolerance: float = 1e-12

    def run(self, problem: Any, result: DiscoveryResult[Any]) -> DistillationResult[Any]:
        hook = getattr(problem, "distill_rational", None)
        if hook is None:
            raise NotImplementedError(
                f"{type(problem).__name__} must implement distill_rational()"
            )
        value = hook(result, max_denominator=self.max_denominator,
                     tolerance=self.tolerance, prune_tolerance=self.prune_tolerance)
        return value if isinstance(value, DistillationResult) else DistillationResult(
            value=value, metadata={"distiller": "rational"}
        )


distillers.register("rational", RationalClusterDistiller)

