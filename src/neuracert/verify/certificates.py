"""Generic certificate envelope and exact-verifier adapter."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping

from neuracert.core.registry import verifiers
from neuracert.core.result import RefinementResult, VerificationResult


@dataclass(frozen=True)
class Certificate:
    format: str
    problem: str
    claim: Mapping[str, Any]
    payload: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()


@dataclass
class ExactCertificateVerifier:
    """Run an injected independent verifier against a refined candidate.

    The callable is injected rather than imported from a problem plugin, which
    allows deployments to keep the verifier implementation dependency-isolated.
    """

    verifier: Callable[[Any], Any]

    def run(self, problem: Any, result: RefinementResult[Any]) -> VerificationResult[Any]:
        outcome = self.verifier(result.value)
        if isinstance(outcome, VerificationResult):
            return outcome
        verified = outcome if isinstance(outcome, bool) else bool(
            getattr(outcome, "verified", False)
        )
        return VerificationResult(value=outcome, verified=verified,
                                  metadata={"verifier": "exact-certificate"})


verifiers.register("exact", ExactCertificateVerifier)

