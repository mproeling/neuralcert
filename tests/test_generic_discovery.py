"""A non-Maynard problem can use generic neural discovery unchanged."""

from __future__ import annotations

import torch

from neuralcert import Evaluation, FunctionalProblem, Grid, SamplingConfig, discover
from neuralcert.discovery import MLPConfig


class QuadraticProblem(FunctionalProblem):
    name = "quadratic-test"

    def sample_domain(self, config: SamplingConfig) -> Grid:
        dtype = getattr(torch, config.dtype)
        x = torch.linspace(-1, 1, config.size, dtype=dtype, device=config.device)
        return Grid(x[:, None])

    def evaluate(self, candidate, grid: Grid) -> Evaluation:
        prediction = candidate(grid.points).squeeze(-1)
        target = grid.points.squeeze(-1).square()
        return Evaluation(-torch.mean((prediction - target).square()))

    def validate(self, candidate, grid: Grid):
        return {"sample_count": len(grid)}


def test_discover_trains_user_problem() -> None:
    result = discover(
        QuadraticProblem(),
        steps=80,
        sample_size=32,
        learning_rate=2e-2,
        model_config=MLPConfig(hidden=(16, 16)),
    )
    assert result.metrics["objective"] > -0.08
    assert result.metadata["model"] == "mlp"
    assert result.diagnostics[0].name == "sample_count"

