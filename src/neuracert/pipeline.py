"""Composable discovery → distillation → refinement → verification pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

from .core.result import (
    Diagnostic,
    DiscoveryResult,
    DistillationResult,
    PipelineResult,
    RefinementResult,
)


@dataclass
class Pipeline:
    discovery: Any
    diagnostics: Any | Sequence[Any] | None = None
    distiller: Any | None = None
    refiner: Any | None = None
    verifier: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.refiner is not None and self.distiller is None:
            raise ValueError("a refiner requires a distiller stage")
        if self.verifier is not None and self.refiner is None:
            raise ValueError("a verifier requires a refiner stage")

    def _diagnostic_components(self) -> tuple[Any, ...]:
        if self.diagnostics is None:
            return ()
        if isinstance(self.diagnostics, Sequence) and not isinstance(
            self.diagnostics, (str, bytes)
        ):
            return tuple(self.diagnostics)
        return (self.diagnostics,)

    def run(self, problem: Any) -> PipelineResult:
        """Execute configured stages in order and retain every intermediate."""
        name = str(getattr(problem, "name", type(problem).__name__))
        output = PipelineResult(
            problem_name=name,
            metadata={"started_at": datetime.now(timezone.utc).isoformat(), **self.metadata},
        )

        discovered = self.discovery.run(problem)
        if not isinstance(discovered, DiscoveryResult):
            discovered = DiscoveryResult(value=discovered)
        for probe in self._diagnostic_components():
            observations = probe.run(problem, discovered)
            if isinstance(observations, list):
                discovered.diagnostics.extend(observations)
        output.discovery = discovered

        if self.distiller is None:
            return output
        distilled = self.distiller.run(problem, discovered)
        if not isinstance(distilled, DistillationResult):
            distilled = DistillationResult(value=distilled)
        output.distillation = distilled

        if self.refiner is None:
            return output
        refined = self.refiner.run(problem, distilled)
        if not isinstance(refined, RefinementResult):
            refined = RefinementResult(value=refined)
        output.refinement = refined

        if self.verifier is not None:
            output.verification = self.verifier.run(problem, refined)
        return output
