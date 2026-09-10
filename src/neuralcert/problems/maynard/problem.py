"""Registration metadata for the existing Maynard implementation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from neuralcert.core.registry import problems
from neuralcert.core.result import DistillationResult


@dataclass(frozen=True)
class MaynardProblem:
    """Configuration handle for the specialized legacy Maynard pipelines.

    Maynard's factored convolution and exact CRT algorithms are not generic
    ``Problem.evaluate`` implementations.  Keeping that distinction explicit
    avoids pretending that specialized certification is interchangeable with a
    neural objective.  The existing ``maynard-*`` CLIs remain the production
    interface while adapters are migrated stage by stage.
    """

    k: int
    epsilon: float = 0.0
    method: str = "poly"
    name: str = "maynard"

    def __post_init__(self) -> None:
        if self.k < 2:
            raise ValueError("k must be at least 2")
        if self.method not in {"poly", "ratio"}:
            raise ValueError("Maynard method must be 'poly' or 'ratio'")
        if not 0 <= self.epsilon < 1:
            raise ValueError("epsilon must lie in [0, 1)")

    def distill_rational(
        self,
        result,
        *,
        max_denominator: int = 10**12,
        tolerance: float = 1e-10,
        prune_tolerance: float = 1e-12,
    ) -> DistillationResult[Any]:
        """Variable-project an exported neural channel onto rational clusters."""
        value = result.value
        if isinstance(value, (str, Path)):
            options: Mapping[str, Any] = {"npz": value}
        elif isinstance(value, Mapping):
            options = value
        else:
            raise TypeError("Maynard rational distillation expects an NPZ path or mapping")
        try:
            path = options["npz"]
        except KeyError as exc:
            raise ValueError("Maynard rational distillation requires an 'npz' path") from exc
        from .distill import distill_neural_npz

        fit, metadata = distill_neural_npz(
            path,
            options.get("multiplicities", options.get("mu", (1,))),
            channel=options.get("channel"),
            initial_poles=options.get("initial_poles"),
            maxiter=int(options.get("maxiter", 300)),
            tolerance=tolerance,
            prune_tolerance=prune_tolerance,
        )
        metadata.update({
            "distiller": "rational-variable-projection",
            "max_denominator": max_denominator,
        })
        return DistillationResult(
            value=fit,
            metrics={"relative_fit_error": fit.relative_error,
                     "design_condition": fit.condition},
            metadata=metadata,
        )


problems.register("maynard", MaynardProblem)
