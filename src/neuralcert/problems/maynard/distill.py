"""Maynard rational distillation and candidate export adapters."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from neuralcert.distill.rational import (
    RationalVariableProjectionFit,
    variable_projection_fit,
)


def trapezoid_weights(points) -> np.ndarray:
    """Positive quadrature weights for a strictly increasing sample grid."""
    t = np.asarray(points, dtype=float).reshape(-1)
    if t.size < 2 or not np.all(np.diff(t) > 0.0):
        raise ValueError("distillation points must be strictly increasing")
    weights = np.empty_like(t)
    weights[0] = 0.5 * (t[1] - t[0])
    weights[-1] = 0.5 * (t[-1] - t[-2])
    if t.size > 2:
        weights[1:-1] = 0.5 * (t[2:] - t[:-2])
    return weights


def load_neural_channel(path: str | Path, channel: int | None = None):
    """Load one scalar neural channel from a polynomial-discovery NPZ."""
    with np.load(path) as data:
        missing = {"k", "x_fine", "g_fine"} - set(data.files)
        if missing:
            raise ValueError(f"neural NPZ missing required fields: {sorted(missing)}")
        k = int(np.asarray(data["k"]).item())
        points = np.asarray(data["x_fine"], dtype=float).reshape(-1)
        values = np.asarray(data["g_fine"], dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        elif values.ndim != 2:
            raise ValueError("g_fine must be a one- or two-dimensional numeric array")
        if values.shape[0] != points.size and values.shape[1] == points.size:
            values = values.T
        if values.shape[0] != points.size:
            raise ValueError("g_fine sample dimension does not match x_fine")
        if channel is None:
            if values.shape[1] == 1:
                channel = 0
            elif "c" in data.files:
                mixing = np.asarray(data["c"], dtype=float).reshape(-1)
                if mixing.size != values.shape[1]:
                    raise ValueError("NPZ mixing vector does not match g_fine channels")
                channel = int(np.argmax(np.abs(mixing)))
            else:
                raise ValueError("multi-channel NPZ requires an explicit channel index")
        if not 0 <= int(channel) < values.shape[1]:
            raise IndexError(f"channel {channel} outside [0, {values.shape[1]})")
        epsilon = float(np.asarray(data["epsilon"]).item()) if "epsilon" in data.files else 0.0
    return k, points, values[:, int(channel)], int(channel), epsilon


def distill_neural_npz(
    path: str | Path,
    multiplicities,
    *,
    channel: int | None = None,
    initial_poles=None,
    maxiter: int = 300,
    tolerance: float = 1e-10,
    prune_tolerance: float = 1e-12,
) -> tuple[RationalVariableProjectionFit, dict[str, Any]]:
    """Fit the paper's clustered rational family to an exported neural channel."""
    k, points, target, selected, epsilon = load_neural_channel(path, channel)
    mu = tuple(int(value) for value in multiplicities)
    if not mu or any(value < 1 for value in mu):
        raise ValueError("multiplicities must be positive")
    if initial_poles is None:
        if k <= 1:
            raise ValueError("Maynard distillation requires k >= 2")
        centre = 1.0 / max(math.log(k - 1) - 0.13, 0.25)
        initial_poles = (centre * np.geomspace(0.25, 4.0, len(mu))
                         if len(mu) > 1 else np.array([centre]))
    fit = variable_projection_fit(
        points,
        target,
        initial_poles=initial_poles,
        multiplicities=mu,
        scale=float(k - 1),
        sample_weights=trapezoid_weights(points),
        maxiter=maxiter,
        tolerance=tolerance,
        prune_tolerance=prune_tolerance,
    )
    metadata = {
        "source": str(path),
        "k": k,
        "epsilon": epsilon,
        "channel": selected,
        "samples": int(points.size),
    }
    return fit, metadata


def export_ratio(*args: Any, **kwargs: Any):
    from maynard_tools.discovery.ratio import export

    return export(*args, **kwargs)
