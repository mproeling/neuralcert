"""Configuration object for finite Delsarte LP problems.

The Delsarte workflow is a finite linear program rather than a neural
functional objective.  This plugin therefore exposes the same registry-level
configuration boundary as Maynard without pretending it implements the
generic differentiable ``Problem.evaluate`` protocol.
"""

from __future__ import annotations

from dataclasses import dataclass

from neuralcert.core.registry import problems
from neuralcert.exact.schemes import AssociationScheme, HammingScheme, JohnsonScheme

from .certify import DualCertificate, discover_and_certify, resolve_constrained


@dataclass(frozen=True)
class DelsarteCodeProblem:
    """A Hamming- or Johnson-scheme code-bound problem."""

    n: int
    min_distance: int
    scheme: str = "hamming"
    q: int = 2
    weight: int | None = None
    spectral_degree: int | None = None
    name: str = "delsarte"

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError("n must be positive")
        if self.scheme not in {"hamming", "johnson"}:
            raise ValueError("scheme must be 'hamming' or 'johnson'")
        if self.scheme == "hamming" and self.q < 2:
            raise ValueError("q must be at least 2")
        if self.scheme == "johnson" and self.weight is None:
            raise ValueError("weight is required for the Johnson scheme")

    def association_scheme(self) -> AssociationScheme:
        if self.scheme == "hamming":
            return HammingScheme(self.n, self.q)
        return JohnsonScheme(self.n, int(self.weight))

    def certify(self, *, bits: int = 50, max_bits: int = 400,
                tolerance: float = 1e-6, verbose: bool = True) -> DualCertificate:
        scheme = self.association_scheme()
        constrained = resolve_constrained(scheme, min_distance=self.min_distance)
        return discover_and_certify(
            scheme,
            constrained,
            spectral_degree=self.spectral_degree,
            min_distance=self.min_distance,
            bits=bits,
            max_bits=max_bits,
            tolerance=tolerance,
            verbose=verbose,
        )


problems.register("delsarte", DelsarteCodeProblem)
