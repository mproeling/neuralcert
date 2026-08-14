"""Deterministic restart policies for non-convex discovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from neuracert.core.result import DiscoveryResult


@dataclass(frozen=True)
class RestartPolicy:
    count: int = 1
    base_seed: int = 0

    def seeds(self) -> Iterable[int]:
        if self.count < 1:
            raise ValueError("restart count must be positive")
        return range(self.base_seed, self.base_seed + self.count)


@dataclass
class RestartedDiscovery:
    """Run independently constructed engines and retain the best objective."""

    factory: Callable[[int], Any]
    policy: RestartPolicy = RestartPolicy()
    maximize: bool = True

    def run(self, problem: Any) -> DiscoveryResult[Any]:
        results = [self.factory(seed).run(problem) for seed in self.policy.seeds()]
        if not results:
            raise RuntimeError("restart policy produced no runs")
        key = lambda result: result.metrics["objective"]
        best = (max if self.maximize else min)(results, key=key)
        best.artifacts["restarts"] = results
        best.metadata["restart_count"] = len(results)
        return best
