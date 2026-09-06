"""Interval-enclosure value objects used by verification plugins."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Generic, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class Interval(Generic[T]):
    lower: T
    upper: T

    def __post_init__(self) -> None:
        if self.lower > self.upper:
            raise ValueError("interval lower bound exceeds upper bound")

    def contains(self, value: T) -> bool:
        return self.lower <= value <= self.upper

    @property
    def width(self):
        return self.upper - self.lower

