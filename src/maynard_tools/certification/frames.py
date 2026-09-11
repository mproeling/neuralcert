"""Shared discovery-export frame conversion for Maynard certifiers.

Discovery builds its Gram matrices from L2-normalised one-dimensional
channels and stores its Ritz vector in a diagonally preconditioned Gram frame.
Certification must reconstruct that same normalised channel frame before an
exported vector can be reused.  Keeping this conversion here prevents exact
backends from silently assigning the right coefficients to the wrong basis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DiscoveryFrame:
    """Numerical discovery data expressed in the normalised channel frame."""

    k: int
    rayleigh: float
    coefficients: np.ndarray
    points: np.ndarray
    channels: np.ndarray
    channel_norms: np.ndarray | None
    logA_diag: np.ndarray | None


def scale_vector_by_logdiag(
    coefficients: np.ndarray,
    log_diagonal: np.ndarray,
    *,
    exponent: float = -0.5,
) -> np.ndarray:
    """Apply a diagonal log-scale without overflow, up to global scale."""
    c = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    ld = np.asarray(log_diagonal, dtype=np.float64).reshape(-1)
    if c.shape != ld.shape:
        raise ValueError("c and logA_diag must have the same shape")
    if not np.all(np.isfinite(c)) or not np.all(np.isfinite(ld)):
        raise ValueError("c and logA_diag must contain only finite values")
    nonzero = c != 0.0
    if not np.any(nonzero):
        raise ValueError("exported c vector is identically zero")
    log_magnitude = np.full_like(c, -np.inf)
    log_magnitude[nonzero] = np.log(np.abs(c[nonzero])) + exponent * ld[nonzero]
    anchor = float(np.max(log_magnitude[nonzero]))
    result = np.zeros_like(c)
    result[nonzero] = np.sign(c[nonzero]) * np.exp(log_magnitude[nonzero] - anchor)
    if not np.all(np.isfinite(result)) or not np.any(result != 0.0):
        raise FloatingPointError("discovery frame conversion underflowed completely")
    return result


def load_discovery_frame(data: Any) -> DiscoveryFrame:
    """Validate an NPZ archive and reconstruct its normalised channel frame.

    ``channel_norms`` scales the exported raw samples only.  It must never be
    multiplied linearly into ``c``.  For projected normalised channels the
    correct coefficient is ``exp(-logA_diag/2) * c_hat`` up to global scale.
    """
    required = {"k", "R", "c", "x_fine", "g_fine"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"discovery NPZ missing fields: {sorted(missing)}")
    k = int(np.asarray(data["k"]).item())
    rayleigh = float(np.asarray(data["R"]).item())
    coefficients = np.asarray(data["c"], dtype=np.float64).reshape(-1)
    points = np.asarray(data["x_fine"], dtype=np.float64).reshape(-1)
    channels = np.asarray(data["g_fine"], dtype=np.float64)
    if channels.ndim == 1:
        channels = channels[:, None]
    if channels.ndim != 2 or channels.shape[0] != points.size:
        raise ValueError("g_fine must have one row per x_fine sample")
    if channels.shape[1] != coefficients.size:
        raise ValueError("c must contain one coefficient per g_fine channel")
    if not (np.all(np.isfinite(points)) and np.all(np.isfinite(channels))):
        raise ValueError("x_fine and g_fine must contain only finite values")

    channel_norms = None
    if "channel_norms" in data.files:
        channel_norms = np.asarray(data["channel_norms"], dtype=np.float64).reshape(-1)
        if channel_norms.shape != coefficients.shape:
            raise ValueError("channel_norms must contain one value per channel")
        if np.any(channel_norms <= 0.0) or not np.all(np.isfinite(channel_norms)):
            raise ValueError("channel_norms must be finite and strictly positive")
        channels = channels / channel_norms[None, :]

    logA_diag = None
    if "logA_diag" in data.files:
        logA_diag = np.asarray(data["logA_diag"], dtype=np.float64).reshape(-1)
        coefficients = scale_vector_by_logdiag(coefficients, logA_diag)
    elif not np.any(coefficients != 0.0):
        raise ValueError("exported c vector is identically zero")

    return DiscoveryFrame(
        k=k,
        rayleigh=rayleigh,
        coefficients=coefficients,
        points=points,
        channels=channels,
        channel_norms=channel_norms,
        logA_diag=logA_diag,
    )
