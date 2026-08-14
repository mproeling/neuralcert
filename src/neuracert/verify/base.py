"""Contracts for independent verification components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from neuracert.core.result import RefinementResult, VerificationResult


class Verifier(Protocol):
    def run(self, problem: Any, result: RefinementResult[Any]) -> VerificationResult[Any]: ...


@dataclass
class CallableVerifier:
    """Adapt an independent callable to the pipeline verifier contract."""

    verify: Callable[[Any], Any]
    claim: str | None = None

    def run(self, problem: Any, result: RefinementResult[Any]) -> VerificationResult[Any]:
        outcome = self.verify(result.value)
        if isinstance(outcome, VerificationResult):
            return outcome
        if isinstance(outcome, bool):
            return VerificationResult(value=outcome, verified=outcome, claim=self.claim)
        verified = bool(getattr(outcome, "verified", False))
        return VerificationResult(value=outcome, verified=verified, claim=self.claim)

