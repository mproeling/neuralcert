"""Pipeline stages remain replaceable and preserve intermediate results."""

from __future__ import annotations

from dataclasses import dataclass

from neuracert import Pipeline
import pytest
from neuracert.core.result import (
    Diagnostic,
    DiscoveryResult,
    DistillationResult,
    RefinementResult,
    VerificationResult,
)


class DemoProblem:
    name = "demo"


class DemoDiscovery:
    def run(self, problem):
        return DiscoveryResult(value=2, metrics={"objective": 2.0})


class DemoProbe:
    def run(self, problem, result):
        return [Diagnostic("positive", result.value > 0, passed=True)]


class DemoDistiller:
    def run(self, problem, result):
        return DistillationResult(value={"exact": result.value})


class DemoRefiner:
    def run(self, problem, result):
        return RefinementResult(value=result.value["exact"] + 1)


class DemoVerifier:
    def run(self, problem, result):
        return VerificationResult(value=result.value, verified=result.value == 3,
                                  claim="demo >= 3")


def test_complete_pipeline() -> None:
    output = Pipeline(
        discovery=DemoDiscovery(),
        diagnostics=DemoProbe(),
        distiller=DemoDistiller(),
        refiner=DemoRefiner(),
        verifier=DemoVerifier(),
    ).run(DemoProblem())
    assert output.verified
    assert output.final is output.verification
    assert output.summary()["stages"] == ["discovery", "distill", "refine", "verify"]
    assert output.discovery.diagnostics[0].name == "positive"


def test_pipeline_rejects_missing_intermediate_stage() -> None:
    with pytest.raises(ValueError, match="refiner requires"):
        Pipeline(discovery=DemoDiscovery(), refiner=DemoRefiner())
    with pytest.raises(ValueError, match="verifier requires"):
        Pipeline(discovery=DemoDiscovery(), verifier=DemoVerifier())
