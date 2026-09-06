"""High-level convenience API."""

from __future__ import annotations

from typing import Any

from .discovery import MLPConfig, NeuralDiscovery, NeuralDiscoveryConfig, OptimizerConfig


def discover(
    problem: Any,
    model: str | Any = "mlp",
    *,
    device: str = "cpu",
    dtype: str = "float64",
    steps: int = 1000,
    sample_size: int = 1024,
    learning_rate: float = 1e-3,
    seed: int = 0,
    maximize: bool = True,
    model_config: MLPConfig | None = None,
    **sampling_options: Any,
):
    """Run generic neural discovery for a user-supplied problem.

    For multi-stage workflows instantiate :class:`neuralcert.Pipeline` instead.
    """
    config = NeuralDiscoveryConfig(
        steps=steps,
        sample_size=sample_size,
        seed=seed,
        device=device,
        dtype=dtype,
        maximize=maximize,
        optimizer=OptimizerConfig(learning_rate=learning_rate),
        sampling_options=sampling_options,
    )
    return NeuralDiscovery(model=model, config=config,
                           model_config=model_config or MLPConfig()).run(problem)

