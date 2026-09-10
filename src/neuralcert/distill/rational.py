"""Variable projection and policy for clustered rational distillation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize

from neuralcert.core.registry import distillers
from neuralcert.core.result import DiscoveryResult, DistillationResult


@dataclass(frozen=True)
class RationalVariableProjectionFit:
    """A fitted clustered rational family and its numerical diagnostics."""

    poles: np.ndarray
    multiplicities: tuple[int, ...]
    coefficients: np.ndarray
    powers: tuple[int, ...]
    retained: np.ndarray
    fitted: np.ndarray
    residual: np.ndarray
    weighted_error: float
    relative_error: float
    rank: int
    condition: float
    iterations: int
    success: bool
    message: str


def clustered_rational_design(
    points: np.ndarray,
    poles: np.ndarray,
    multiplicities: tuple[int, ...] | list[int],
    scale: float,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Construct columns ``(c_a + scale*t)^(-j)`` in cluster order."""
    t = np.asarray(points, dtype=float).reshape(-1)
    c = np.asarray(poles, dtype=float).reshape(-1)
    mu = tuple(int(value) for value in multiplicities)
    if len(c) != len(mu) or not len(c):
        raise ValueError("poles and multiplicities must have the same nonzero length")
    if np.any(c <= 0.0) or not np.all(np.isfinite(c)):
        raise ValueError("all pole locations must be finite and positive")
    if any(value < 1 for value in mu):
        raise ValueError("all multiplicities must be positive")
    if scale <= 0.0 or not math.isfinite(scale):
        raise ValueError("scale must be finite and positive")
    denominators = c[:, None] + scale * t[None, :]
    if np.any(denominators <= 0.0):
        raise ValueError("rational basis has a pole on the sampled domain")
    powers = tuple(power for multiplicity in mu for power in range(1, multiplicity + 1))
    columns = [denominators[index] ** (-power)
               for index, multiplicity in enumerate(mu)
               for power in range(1, multiplicity + 1)]
    return np.stack(columns, axis=1), powers


def _validate_samples(points, target, sample_weights):
    t = np.asarray(points, dtype=float).reshape(-1)
    y = np.asarray(target, dtype=float).reshape(-1)
    if t.shape != y.shape or t.size < 2:
        raise ValueError("points and target must be one-dimensional arrays of equal length")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)):
        raise ValueError("points and target must be finite")
    if sample_weights is None:
        omega = np.ones_like(t)
    else:
        omega = np.asarray(sample_weights, dtype=float).reshape(-1)
        if omega.shape != t.shape:
            raise ValueError("sample_weights must match points")
    if np.any(omega < 0.0) or not np.all(np.isfinite(omega)) or not np.any(omega > 0.0):
        raise ValueError("sample_weights must be finite, nonnegative and not all zero")
    return t, y, omega


def variable_project_coefficients(
    points,
    target,
    poles,
    multiplicities,
    *,
    scale: float,
    sample_weights=None,
    rcond: float | None = None,
):
    """Solve the weighted linear least-squares problem for fixed poles.

    Column equilibration is undone after the solve, so the returned
    coefficients multiply the physical basis functions from the paper.
    """
    t, y, omega = _validate_samples(points, target, sample_weights)
    design, powers = clustered_rational_design(t, poles, multiplicities, scale)
    root_weight = np.sqrt(omega)
    weighted_design = root_weight[:, None] * design
    weighted_target = root_weight * y
    column_norm = np.linalg.norm(weighted_design, axis=0)
    if np.any(column_norm == 0.0) or not np.all(np.isfinite(column_norm)):
        raise np.linalg.LinAlgError("rational design contains a zero or non-finite column")
    equilibrated = weighted_design / column_norm
    scaled_coefficients, _, rank, singular_values = np.linalg.lstsq(
        equilibrated, weighted_target, rcond=rcond
    )
    coefficients = scaled_coefficients / column_norm
    fitted = design @ coefficients
    residual = y - fitted
    weighted_error = float(np.dot(omega, residual * residual))
    target_energy = float(np.dot(omega, y * y))
    relative_error = math.sqrt(weighted_error / max(target_energy, np.finfo(float).tiny))
    condition = (float(singular_values[0] / singular_values[-1])
                 if singular_values.size and singular_values[-1] > 0.0 else math.inf)
    return {
        "coefficients": coefficients,
        "powers": powers,
        "fitted": fitted,
        "residual": residual,
        "weighted_error": weighted_error,
        "relative_error": relative_error,
        "rank": int(rank),
        "condition": condition,
        "contributions": np.abs(scaled_coefficients),
    }


def _poles_to_unconstrained(poles: np.ndarray) -> np.ndarray:
    logs = np.log(np.sort(np.asarray(poles, dtype=float)))
    if len(logs) == 1:
        return logs
    gaps = np.diff(logs)
    if np.any(gaps <= 0.0):
        raise ValueError("initial pole locations must be strictly increasing")
    return np.concatenate([[logs[0]], np.log(np.expm1(gaps))])


def _unconstrained_to_poles(parameters: np.ndarray) -> np.ndarray:
    x = np.asarray(parameters, dtype=float)
    logs = np.concatenate([[x[0]], x[0] + np.cumsum(np.logaddexp(0.0, x[1:]))])
    return np.exp(logs)


def variable_projection_fit(
    points,
    target,
    *,
    initial_poles,
    multiplicities,
    scale: float,
    sample_weights=None,
    maxiter: int = 300,
    tolerance: float = 1e-10,
    prune_tolerance: float = 1e-12,
    rcond: float | None = None,
) -> RationalVariableProjectionFit:
    """Optimize only pole locations, projecting linear coefficients at each step."""
    t, y, omega = _validate_samples(points, target, sample_weights)
    mu = tuple(int(value) for value in multiplicities)
    x0 = _poles_to_unconstrained(np.asarray(initial_poles, dtype=float))
    target_energy = max(float(np.dot(omega, y * y)), np.finfo(float).tiny)

    def objective(parameters):
        try:
            projection = variable_project_coefficients(
                t, y, _unconstrained_to_poles(parameters), mu,
                scale=scale, sample_weights=omega, rcond=rcond,
            )
            return projection["weighted_error"] / target_energy
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            return np.finfo(float).max ** 0.25

    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        options={"maxiter": int(maxiter), "ftol": float(tolerance), "gtol": 1e-10},
    )
    poles = _unconstrained_to_poles(result.x)
    projection = variable_project_coefficients(
        t, y, poles, mu, scale=scale, sample_weights=omega, rcond=rcond,
    )
    contributions = projection["contributions"]
    threshold = prune_tolerance * max(float(np.max(contributions)), np.finfo(float).tiny)
    retained = contributions >= threshold
    powers = projection["powers"]
    inferred = []
    offset = 0
    for multiplicity in mu:
        kept_powers = [powers[offset + index] for index in range(multiplicity)
                       if retained[offset + index]]
        inferred.append(max(kept_powers, default=0))
        offset += multiplicity
    return RationalVariableProjectionFit(
        poles=poles,
        multiplicities=tuple(inferred),
        coefficients=projection["coefficients"],
        powers=powers,
        retained=retained,
        fitted=projection["fitted"],
        residual=projection["residual"],
        weighted_error=projection["weighted_error"],
        relative_error=projection["relative_error"],
        rank=projection["rank"],
        condition=projection["condition"],
        iterations=int(result.nit),
        success=bool(result.success),
        message=str(result.message),
    )


@dataclass
class RationalClusterDistiller:
    """Request a problem-specific rational representation.

    Rational approximation is domain-sensitive, so this generic component
    supplies policy while the problem owns the mathematical conversion.
    """

    max_denominator: int = 10**12
    tolerance: float = 1e-10
    prune_tolerance: float = 1e-12

    def run(self, problem: Any, result: DiscoveryResult[Any]) -> DistillationResult[Any]:
        hook = getattr(problem, "distill_rational", None)
        if hook is None:
            raise NotImplementedError(
                f"{type(problem).__name__} must implement distill_rational()"
            )
        value = hook(result, max_denominator=self.max_denominator,
                     tolerance=self.tolerance, prune_tolerance=self.prune_tolerance)
        return value if isinstance(value, DistillationResult) else DistillationResult(
            value=value, metadata={"distiller": "rational"}
        )


distillers.register("rational", RationalClusterDistiller)
