"""Optimizer configuration independent of concrete problems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class OptimizerConfig:
    name: str = "adam"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    options: dict[str, Any] | None = None


def build_optimizer(parameters: Iterable[Any], config: OptimizerConfig):
    import torch.optim as optim

    options = dict(config.options or {})
    common = {"lr": config.learning_rate, "weight_decay": config.weight_decay,
              **options}
    name = config.name.lower()
    if name == "adam":
        return optim.Adam(parameters, **common)
    if name == "adamw":
        return optim.AdamW(parameters, **common)
    if name == "sgd":
        return optim.SGD(parameters, **common)
    raise ValueError(f"unsupported optimizer {config.name!r}")
