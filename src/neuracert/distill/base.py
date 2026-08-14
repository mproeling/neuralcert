"""Interfaces for converting numerical candidates to structured forms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from neuracert.core.result import DiscoveryResult, DistillationResult


class Distiller(Protocol):
    def run(self, problem: Any, result: DiscoveryResult[Any]) -> DistillationResult[Any]: ...


@dataclass
class ProblemDistiller:
    """Delegate to a problem's optional, explicitly implemented hook."""

    def run(self, problem: Any, result: DiscoveryResult[Any]) -> DistillationResult[Any]:
        value = problem.distill(result)
        return value if isinstance(value, DistillationResult) else DistillationResult(value=value)

