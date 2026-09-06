"""Registration metadata for the Cohn-Gonçalves sign-uncertainty problem."""

from __future__ import annotations

from dataclasses import dataclass

from neuralcert.core.registry import problems


@dataclass(frozen=True)
class SignUncertaintyProblem:
    """Configuration for a radial Fourier sign-uncertainty computation."""

    dimension: int
    sign: int = 1
    family: str = "laguerre"
    name: str = "sign_uncertainty"

    def __post_init__(self) -> None:
        if self.dimension < 1:
            raise ValueError("dimension must be positive")
        if self.sign not in {-1, 1}:
            raise ValueError("sign must be -1 or +1")
        if self.family not in {"laguerre", "gaussian-mixture", "hybrid"}:
            raise ValueError("family must be laguerre, gaussian-mixture or hybrid")


problems.register("sign_uncertainty", SignUncertaintyProblem)
problems.register("gc", SignUncertaintyProblem)
