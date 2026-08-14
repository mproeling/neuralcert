"""Registration metadata for the existing Maynard implementation."""

from __future__ import annotations

from dataclasses import dataclass

from neuracert.core.registry import problems


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


problems.register("maynard", MaynardProblem)

