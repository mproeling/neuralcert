"""Numerical-to-structured candidate conversion."""

from .base import Distiller, ProblemDistiller
from .rational import (
    RationalClusterDistiller,
    RationalVariableProjectionFit,
    clustered_rational_design,
    variable_project_coefficients,
    variable_projection_fit,
)
from .sparse_basis import SparseBasisDistiller
from .spectral import SpectralDistiller

__all__ = [name for name in globals() if not name.startswith("_")]
