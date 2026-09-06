"""Composable differentiable and non-differentiable constraints."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence


@dataclass
class ConstraintResult:
    name: str
    satisfied: bool
    penalty: Any = 0.0
    value: Any = None
    message: str = ""


class Constraint(Protocol):
    name: str

    def evaluate(self, candidate: Any, grid: Any) -> ConstraintResult: ...


@dataclass
class ConstraintSet:
    constraints: Sequence[Constraint] = field(default_factory=tuple)

    def evaluate(self, candidate: Any, grid: Any) -> list[ConstraintResult]:
        return [constraint.evaluate(candidate, grid) for constraint in self.constraints]

    def penalty(self, candidate: Any, grid: Any) -> Any:
        values = [result.penalty for result in self.evaluate(candidate, grid)]
        return sum(values) if values else 0.0

