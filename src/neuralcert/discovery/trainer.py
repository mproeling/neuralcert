"""Problem-neutral differentiable neural discovery engine."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from neuralcert.core.grids import Grid, SamplingConfig
from neuralcert.core.problem import Evaluation
from neuralcert.core.registry import discovery_engines, models
from neuralcert.core.result import Diagnostic, DiscoveryResult

from .models import MLPConfig  # registers the built-in MLP
from .optimizers import OptimizerConfig, build_optimizer


class DiscoveryEngine(Protocol):
    def run(self, problem: Any) -> DiscoveryResult[Any]: ...


@dataclass(frozen=True)
class NeuralDiscoveryConfig:
    steps: int = 1000
    sample_size: int = 1024
    seed: int = 0
    device: str = "cpu"
    dtype: str = "float64"
    maximize: bool = True
    log_every: int = 100
    resample_every: int = 0
    gradient_clip: float | None = None
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    sampling_options: Mapping[str, Any] = field(default_factory=dict)


def _scalar(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu().item()
    return float(value)


@dataclass
class NeuralDiscovery:
    """Optimize a registered neural model against ``Problem.evaluate``."""

    model: str | Any = "mlp"
    config: NeuralDiscoveryConfig = field(default_factory=NeuralDiscoveryConfig)
    model_config: Any = None

    def _sample(self, problem: Any, seed: int) -> Grid[Any]:
        return problem.sample_domain(SamplingConfig(
            size=self.config.sample_size,
            seed=seed,
            device=self.config.device,
            dtype=self.config.dtype,
            options=self.config.sampling_options,
        ))

    def _build_model(self, grid: Grid[Any]):
        import torch

        if isinstance(self.model, str):
            shape = getattr(grid.points, "shape", None)
            if shape is None or len(shape) < 2:
                raise ValueError("model input dimension cannot be inferred from grid.points")
            candidate = models.create(
                self.model,
                input_dim=int(shape[-1]),
                config=self.model_config or MLPConfig(),
            )
        else:
            candidate = self.model
        dtype = getattr(torch, self.config.dtype, None)
        if dtype is None:
            raise ValueError(f"unknown torch dtype {self.config.dtype!r}")
        return candidate.to(device=self.config.device, dtype=dtype)

    def run(self, problem: Any) -> DiscoveryResult[Any]:
        import torch

        if self.config.steps < 1:
            raise ValueError("steps must be positive")
        torch.manual_seed(self.config.seed)
        grid = self._sample(problem, self.config.seed).to(self.config.device)
        candidate = self._build_model(grid)
        optimizer = build_optimizer(candidate.parameters(), self.config.optimizer)
        history: list[float] = []
        best_score = -math.inf if self.config.maximize else math.inf
        best_state: dict[str, Any] | None = None
        best_metrics: dict[str, float] = {}

        for step in range(self.config.steps):
            if self.config.resample_every and step and step % self.config.resample_every == 0:
                grid = self._sample(problem, self.config.seed + step).to(self.config.device)
            optimizer.zero_grad()
            evaluated = problem.evaluate(candidate, grid)
            evaluation = evaluated if isinstance(evaluated, Evaluation) else Evaluation(evaluated)
            objective = evaluation.objective
            if not hasattr(objective, "backward") or objective.numel() != 1:
                raise TypeError("Problem.evaluate must return a scalar differentiable tensor")
            penalty = problem.constraints(candidate, grid) if hasattr(problem, "constraints") else 0.0
            loss = (-objective if self.config.maximize else objective) + penalty
            loss.backward()
            if self.config.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(candidate.parameters(), self.config.gradient_clip)
            optimizer.step()

            score = _scalar(objective)
            history.append(score)
            improved = score > best_score if self.config.maximize else score < best_score
            if improved and math.isfinite(score):
                best_score = score
                best_state = copy.deepcopy(candidate.state_dict())
                best_metrics = {
                    str(name): _scalar(value)
                    for name, value in evaluation.metrics.items()
                }

        if best_state is None:
            raise FloatingPointError("discovery produced no finite objective")
        candidate.load_state_dict(best_state)
        raw_diagnostics = problem.validate(candidate, grid) if hasattr(problem, "validate") else {}
        diagnostics = [Diagnostic(str(key), value) for key, value in raw_diagnostics.items()]
        return DiscoveryResult(
            value=candidate,
            metrics={"objective": best_score, "steps": float(self.config.steps),
                     **best_metrics},
            diagnostics=diagnostics,
            artifacts={"grid": grid, "history": history, "state_dict": best_state},
            metadata={"engine": "neural", "model": self.model if isinstance(self.model, str)
                      else type(self.model).__name__, "device": self.config.device,
                      "dtype": self.config.dtype},
        )


discovery_engines.register("neural", NeuralDiscovery)
