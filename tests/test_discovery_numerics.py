"""Fast smoke tests for extracted discovery primitives."""

from __future__ import annotations

import numpy as np

from maynard_tools.discovery.numerics import gauss_legendre_01, linear_interp_pairs


def test_gauss_legendre_integrates_cubic() -> None:
    nodes, weights = gauss_legendre_01(3)
    assert np.isclose(np.dot(weights, nodes**3), 0.25)


def test_linear_interpolation_hits_nodes() -> None:
    nodes = np.array([0.0, 0.5, 1.0])
    left, right, weight_left, weight_right = linear_interp_pairs(nodes, nodes)
    reconstructed = weight_left * nodes[left] + weight_right * nodes[right]
    np.testing.assert_allclose(reconstructed, nodes)
