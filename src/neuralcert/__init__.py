"""NeuralCert: reusable numerical discovery and exact verification pipelines."""

from .api import discover
from .core import Evaluation, FunctionalProblem, Grid, PipelineResult, Problem, SamplingConfig
from .pipeline import Pipeline
from . import problems as _builtin_problems  # register bundled problem plugins

__version__ = "0.12.3"

__all__ = [
    "Evaluation",
    "FunctionalProblem",
    "Grid",
    "Pipeline",
    "PipelineResult",
    "Problem",
    "SamplingConfig",
    "discover",
]
