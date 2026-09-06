"""Generic candidate models and the public model registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from neuralcert.core.registry import models


@dataclass(frozen=True)
class MLPConfig:
    hidden: tuple[int, ...] = (64, 64, 64)
    output_dim: int = 1
    activation: str = "tanh"


def build_mlp(input_dim: int, config: MLPConfig | None = None, **_: Any):
    """Create a plain scalar/vector MLP without problem-specific transforms."""
    import torch.nn as nn

    cfg = config or MLPConfig()
    activations = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU,
                   "silu": nn.SiLU}
    try:
        activation = activations[cfg.activation.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported activation {cfg.activation!r}") from exc
    widths = (input_dim, *cfg.hidden, cfg.output_dim)
    layers: list[nn.Module] = []
    for left, right in zip(widths[:-2], widths[1:-1]):
        layers.extend((nn.Linear(left, right), activation()))
    layers.append(nn.Linear(widths[-2], widths[-1]))
    return nn.Sequential(*layers)


models.register("mlp", build_mlp)

