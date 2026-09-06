"""Problem-neutral domain sampling containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Iterator, Mapping, TypeVar


ArrayT = TypeVar("ArrayT")


@dataclass(frozen=True)
class SamplingConfig:
    """Generic sampling request; problems may interpret ``options`` freely."""

    size: int = 1024
    seed: int = 0
    device: str = "cpu"
    dtype: str = "float64"
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class Grid(Generic[ArrayT]):
    """Points, optional weights and metadata supplied to a problem."""

    points: ArrayT
    weights: ArrayT | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to(self, device: str) -> "Grid[Any]":
        """Move tensor-like contents while remaining NumPy-compatible."""
        points = self.points.to(device) if hasattr(self.points, "to") else self.points
        weights = self.weights
        if weights is not None and hasattr(weights, "to"):
            weights = weights.to(device)
        return Grid(points=points, weights=weights, metadata=dict(self.metadata))

    def __len__(self) -> int:
        return len(self.points)  # type: ignore[arg-type]

