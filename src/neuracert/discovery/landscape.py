"""Optional probes that inspect a discovered candidate without changing it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from neuracert.core.result import Diagnostic, DiscoveryResult


@dataclass
class LandscapeProbe:
    """Collect problem-defined diagnostics at the discovery grid."""

    name: str = "landscape"

    def run(self, problem: Any, result: DiscoveryResult[Any]) -> list[Diagnostic]:
        grid = result.artifacts.get("grid")
        if grid is None or not hasattr(problem, "diagnostics"):
            return []
        values: Mapping[str, Any] = problem.diagnostics(result.value, grid)
        return [Diagnostic(name=str(key), value=value) for key, value in values.items()]

