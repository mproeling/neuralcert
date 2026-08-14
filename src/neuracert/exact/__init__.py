"""Reusable exact-arithmetic and association-scheme infrastructure.

This layer is problem-independent. Problem plugins may use it to construct
certificates, while independent verifiers deliberately must not import it.
"""

from .crt import ModularProblem, MultiModularCertifier
from .schemes import AssociationScheme, ExplicitScheme, HammingScheme, JohnsonScheme

__all__ = [
    "AssociationScheme",
    "ExplicitScheme",
    "HammingScheme",
    "JohnsonScheme",
    "ModularProblem",
    "MultiModularCertifier",
]
