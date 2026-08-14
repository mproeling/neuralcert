"""Typed results passed between NeuraCert pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, Mapping, TypeVar


T = TypeVar("T")


class Stage(str, Enum):
    DISCOVERY = "discovery"
    DISTILL = "distill"
    REFINE = "refine"
    VERIFY = "verify"


@dataclass(frozen=True)
class Diagnostic:
    """One machine-readable validation or landscape observation."""

    name: str
    value: Any
    passed: bool | None = None
    message: str = ""


@dataclass
class StageResult(Generic[T]):
    """Output of one stage, including diagnostics and reproducibility data."""

    value: T
    metrics: dict[str, float] = field(default_factory=dict)
    diagnostics: list[Diagnostic] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DiscoveryResult(StageResult[T]):
    """Numerical candidate produced by a discovery engine."""


@dataclass
class DistillationResult(StageResult[T]):
    """Structured or exact-friendly representation of a candidate."""


@dataclass
class RefinementResult(StageResult[T]):
    """Locally or algebraically improved distilled candidate."""


@dataclass
class VerificationResult(StageResult[T]):
    """Independent verification outcome."""

    verified: bool = False
    claim: str | None = None


@dataclass
class PipelineResult:
    """Complete immutable-by-convention record of a pipeline run."""

    problem_name: str
    discovery: DiscoveryResult[Any] | None = None
    distillation: DistillationResult[Any] | None = None
    refinement: RefinementResult[Any] | None = None
    verification: VerificationResult[Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def final(self) -> StageResult[Any] | None:
        return self.verification or self.refinement or self.distillation or self.discovery

    @property
    def verified(self) -> bool:
        return bool(self.verification and self.verification.verified)

    def summary(self) -> Mapping[str, Any]:
        return {
            "problem": self.problem_name,
            "stages": [
                stage.value
                for stage, result in (
                    (Stage.DISCOVERY, self.discovery),
                    (Stage.DISTILL, self.distillation),
                    (Stage.REFINE, self.refinement),
                    (Stage.VERIFY, self.verification),
                )
                if result is not None
            ],
            "verified": self.verified,
        }

