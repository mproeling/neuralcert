"""Regression tests for ratio discovery v8 and its epsilon extension."""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from maynard_tools.certification.ratio import load
from maynard_tools.discovery.ratio import Grid, _cf_simple, export, rayleigh


def test_characteristic_function_uses_enlarged_interval() -> None:
    c, n, epsilon = 0.2, 50, 0.03
    value = _cf_simple(np.array([0.0]), c, n, 1.0 + epsilon)[0].real
    expected = np.log1p(n * (1.0 + epsilon) / c) / n
    assert np.isclose(value, expected, rtol=0, atol=1e-15)


def test_epsilon_changes_the_rayleigh_problem() -> None:
    grid = Grid(N=1 << 14)
    ordinary = rayleigh([0.2], [1], 51, grid, eps=0.0)[0]
    enlarged = rayleigh([0.2], [1], 51, grid, eps=0.01)[0]
    assert np.isfinite(enlarged)
    assert enlarged <= 51
    assert abs(enlarged - ordinary) > 1e-4


def test_epsilon_is_hashed_and_exported_for_certification_guard(tmp_path) -> None:
    grid = Grid(N=1 << 13)
    value, vector, _ = rayleigh([0.2], [1], 51, grid, eps=0.01)
    path = tmp_path / "epsilon.npz"
    export(path, 51, [1], [0.2], vector, value, grid, eps=0.01)
    with np.load(path) as archive:
        assert all(archive[name].dtype.kind != "O" for name in archive.files)
    document = load(path)
    assert document["epsilon"] == Fraction(1, 100)
    assert "epsilon=1/100" in document["canonical"]
