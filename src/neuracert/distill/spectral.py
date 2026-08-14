"""Spectral distillation policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from neuracert.core.result import DiscoveryResult, DistillationResult
from neuracert.core.registry import distillers


@dataclass
class SpectralDistiller:
    degree: int = 32
    basis: str = "chebyshev"

    def run(self, problem: Any, result: DiscoveryResult[Any]) -> DistillationResult[Any]:
        hook = getattr(problem, "distill_spectral", None)
        if hook is None:
            raise NotImplementedError(
                f"{type(problem).__name__} must implement distill_spectral()"
            )
        value = hook(result, degree=self.degree, basis=self.basis)
        return value if isinstance(value, DistillationResult) else DistillationResult(
            value=value, metadata={"distiller": "spectral", "basis": self.basis}
        )


distillers.register("spectral", SpectralDistiller)
