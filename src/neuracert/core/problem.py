"""Contracts implemented by number-theory problem plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

from .grids import Grid, SamplingConfig


@dataclass
class Evaluation:
    """Differentiable scalar objective plus auxiliary values."""

    objective: Any
    metrics: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class Problem(Protocol):
    """Minimal interface required by generic discovery."""

    name: str

    def sample_domain(self, config: SamplingConfig) -> Grid[Any]: ...

    def evaluate(self, candidate: Any, grid: Grid[Any]) -> Evaluation | Any: ...

    def validate(self, candidate: Any, grid: Grid[Any]) -> Mapping[str, Any]: ...


class FunctionalProblem(ABC):
    """Convenience base class with optional pipeline extension hooks.

    Only sampling and evaluation are mandatory.  Distillation, refinement and
    verification remain algorithmic components; hooks provide problem-specific
    defaults without forcing every problem to implement the entire pipeline.
    """

    name = "functional-problem"

    @abstractmethod
    def sample_domain(self, config: SamplingConfig) -> Grid[Any]:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, candidate: Any, grid: Grid[Any]) -> Evaluation | Any:
        raise NotImplementedError

    def validate(self, candidate: Any, grid: Grid[Any]) -> Mapping[str, Any]:
        return {}

    def transform_input(self, points: Any) -> Any:
        """Optional coordinate transform used by problem implementations."""
        return points

    def reference_function(self, points: Any) -> Any:
        """Optional analytic/reference candidate for controls and diagnostics."""
        raise NotImplementedError(f"{type(self).__name__} has no reference function")

    def constraints(self, candidate: Any, grid: Grid[Any]) -> Any:
        return 0.0

    def diagnostics(self, candidate: Any, grid: Grid[Any]) -> Mapping[str, Any]:
        return self.validate(candidate, grid)

    def distill(self, discovery_result: Any) -> Any:
        raise NotImplementedError(f"{type(self).__name__} has no default distiller")

    def refine(self, distilled: Any) -> Any:
        raise NotImplementedError(f"{type(self).__name__} has no default refiner")

    def verify(self, refined: Any) -> Any:
        raise NotImplementedError(f"{type(self).__name__} has no default verifier")
