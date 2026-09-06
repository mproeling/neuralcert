"""Minimal user-defined problem for the generic NeuralCert discovery API."""

from __future__ import annotations

import torch

from neuralcert import Evaluation, FunctionalProblem, Grid, SamplingConfig, discover


class SineApproximationProblem(FunctionalProblem):
    """Fit sin(pi x); replace this objective with number-theoretic functionals."""

    name = "sine-approximation"

    def sample_domain(self, config: SamplingConfig) -> Grid:
        generator = torch.Generator(device="cpu").manual_seed(config.seed)
        dtype = getattr(torch, config.dtype)
        points = torch.rand((config.size, 1), generator=generator, dtype=dtype)
        return Grid(points=points.to(config.device))

    def evaluate(self, candidate, grid: Grid) -> Evaluation:
        predicted = candidate(grid.points).squeeze(-1)
        target = torch.sin(torch.pi * grid.points.squeeze(-1))
        mse = torch.mean((predicted - target) ** 2)
        return Evaluation(objective=-mse, metrics={"mse": mse})

    def validate(self, candidate, grid: Grid):
        with torch.no_grad():
            value = self.evaluate(candidate, grid).objective
        return {"finite": bool(torch.isfinite(value))}


if __name__ == "__main__":
    result = discover(SineApproximationProblem(), steps=500, device="cpu")
    print(result.metrics)

