"""Variable-projection tests for clustered Maynard ratio distillation."""

from __future__ import annotations

import numpy as np
import pytest
import subprocess
import sys
import math

from neuralcert.core.result import DiscoveryResult
from neuralcert.distill import RationalClusterDistiller
from neuralcert.distill.rational import (
    clustered_rational_design,
    variable_project_coefficients,
    variable_projection_fit,
)
from neuralcert.problems.maynard import MaynardProblem
from neuralcert.problems.maynard.distill import load_neural_channel, trapezoid_weights
from maynard_tools.discovery.ratio import ceiling


def test_ratio_ceiling_matches_vanilla_and_epsilon_bounds() -> None:
    k = 50
    assert ceiling(k, 0.0) == pytest.approx(k / (k - 1.0) * math.log(k))
    assert ceiling(k, 0.01) == pytest.approx(
        k / (k - 1.0) * math.log(2.0 * k - 1.0)
    )


def test_fixed_poles_project_linear_coefficients() -> None:
    points = np.linspace(0.0, 1.0, 401)
    poles = np.array([0.2, 0.75])
    multiplicities = (2, 1)
    design, powers = clustered_rational_design(points, poles, multiplicities, 100.0)
    expected = np.array([1.25, -0.08, 0.4])
    target = design @ expected

    projection = variable_project_coefficients(
        points,
        target,
        poles,
        multiplicities,
        scale=100.0,
        sample_weights=trapezoid_weights(points),
    )

    assert powers == (1, 2, 1)
    assert projection["coefficients"] == pytest.approx(expected, rel=1e-10, abs=1e-10)
    assert projection["relative_error"] < 1e-12
    assert projection["rank"] == 3


def test_outer_fit_moves_only_poles_and_recovers_structure() -> None:
    points = np.unique(np.concatenate(([0.0], np.geomspace(1e-5, 1.0, 600))))
    expected_poles = np.array([0.18, 0.62])
    design, _ = clustered_rational_design(points, expected_poles, (2, 1), 50.0)
    target = design @ np.array([1.0, -0.025, 0.35])

    fit = variable_projection_fit(
        points,
        target,
        initial_poles=[0.14, 0.8],
        multiplicities=(2, 1),
        scale=50.0,
        sample_weights=trapezoid_weights(points),
        maxiter=500,
        prune_tolerance=1e-14,
    )

    assert fit.relative_error < 1e-6
    assert fit.poles == pytest.approx(expected_poles, rel=2e-2)
    assert fit.multiplicities == (2, 1)
    assert fit.retained.tolist() == [True, True, True]


def test_npz_channel_selection_and_generic_distiller(tmp_path) -> None:
    points = np.linspace(0.0, 1.0, 301)
    design, _ = clustered_rational_design(points, np.array([0.25]), (1,), 50.0)
    channels = np.column_stack([np.ones_like(points), design[:, 0]])
    path = tmp_path / "neural.npz"
    np.savez(path, k=51, epsilon=0.0, x_fine=points, g_fine=channels,
             c=np.array([0.1, 2.0]))

    k, loaded_points, target, selected, epsilon = load_neural_channel(path)
    assert (k, selected, epsilon) == (51, 1, 0.0)
    assert loaded_points == pytest.approx(points)
    assert target == pytest.approx(channels[:, 1])

    result = RationalClusterDistiller(prune_tolerance=1e-14).run(
        MaynardProblem(k=51, method="ratio"),
        DiscoveryResult(value={"npz": path, "mu": (1,), "initial_poles": [0.2]}),
    )
    assert result.value.relative_error < 1e-7
    assert result.value.poles == pytest.approx([0.25], rel=1e-3)
    assert result.metadata["channel"] == 1


def test_ratio_cli_requires_explicit_neural_opt() -> None:
    command = [
        sys.executable, "-m", "neuralcert.cli", "discover", "--method", "ratio",
        "--k", "51", "--distill-npz", "candidate.npz",
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    assert completed.returncode == 2
    assert "--distill-npz is only used with --opt neural" in completed.stderr

    command = [
        sys.executable, "-m", "neuralcert.cli", "discover", "--method", "ratio",
        "--opt", "neural", "--k", "51",
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    assert completed.returncode == 2
    assert "--opt neural requires --distill-npz FILE" in completed.stderr
