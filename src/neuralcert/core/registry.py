"""Small explicit registries for plugins and pipeline components."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, Generic, TypeVar


T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str, item: T | None = None, *, replace: bool = False):
        def add(value: T) -> T:
            key = name.strip().lower()
            if not key:
                raise ValueError(f"{self.kind} name cannot be empty")
            if key in self._items and not replace:
                raise KeyError(f"{self.kind} {key!r} is already registered")
            self._items[key] = value
            return value

        return add if item is None else add(item)

    def get(self, name: str) -> T:
        key = name.strip().lower()
        try:
            return self._items[key]
        except KeyError as exc:
            choices = ", ".join(sorted(self._items)) or "(none)"
            raise KeyError(f"unknown {self.kind} {name!r}; available: {choices}") from exc

    def create(self, name: str, **kwargs: Any) -> Any:
        item = self.get(name)
        return item(**kwargs) if callable(item) else item

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name.lower() in self._items


problems: Registry[Any] = Registry("problem")
models: Registry[Any] = Registry("model")
discovery_engines: Registry[Any] = Registry("discovery engine")
distillers: Registry[Any] = Registry("distiller")
refiners: Registry[Any] = Registry("refiner")
verifiers: Registry[Any] = Registry("verifier")

