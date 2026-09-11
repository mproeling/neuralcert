"""Regression tests for discovery-to-certification channel frames."""

from __future__ import annotations

import numpy as np

from maynard_tools.certification.frames import load_discovery_frame


def _quotient(matrix_a, matrix_b, vector):
    return float(vector @ matrix_b @ vector / (vector @ matrix_a @ vector))


def test_export_frame_conversion_is_quotient_invariant(tmp_path) -> None:
    points = np.linspace(0.0, 1.0, 9)
    normalised = np.column_stack((1.0 + points, 1.0 - 0.25 * points, points**2 + 0.2))
    norms = np.array([2.0**-7, 2.0**3, 1.25])
    raw = normalised * norms[None, :]
    log_diagonal = np.array([-4.0, 2.0, 0.5])
    c_hat = np.array([0.3, -0.7, 1.1])
    path = tmp_path / "frame.npz"
    np.savez(path, k=11, R=3.5, c=c_hat, x_fine=points, g_fine=raw,
             channel_norms=norms, logA_diag=log_diagonal)

    with np.load(path) as archive:
        frame = load_discovery_frame(archive)

    expected = c_hat * np.exp(-0.5 * log_diagonal)
    expected /= np.max(np.abs(expected))
    assert np.allclose(frame.channels, normalised)
    assert np.allclose(frame.coefficients, expected)
    assert not np.allclose(frame.coefficients, expected * norms)

    a_norm = np.array([[2.0, 0.2, 0.1], [0.2, 1.5, -0.1], [0.1, -0.1, 1.2]])
    b_norm = np.array([[0.8, 0.05, 0.02], [0.05, 0.7, 0.03], [0.02, 0.03, 0.6]])
    root_d = np.diag(np.exp(0.5 * log_diagonal))
    a_stored = np.linalg.inv(root_d) @ a_norm @ np.linalg.inv(root_d)
    b_stored = np.linalg.inv(root_d) @ b_norm @ np.linalg.inv(root_d)
    assert np.isclose(
        _quotient(a_stored, b_stored, c_hat),
        _quotient(a_norm, b_norm, frame.coefficients),
    )


def test_channel_norms_never_change_normalised_frame_coefficients(tmp_path) -> None:
    points = np.array([0.0, 1.0])
    path = tmp_path / "norms.npz"
    np.savez(path, k=5, R=1.0, c=[1.0, 2.0], x_fine=points,
             g_fine=[[3.0, 40.0], [6.0, 80.0]], channel_norms=[3.0, 20.0],
             logA_diag=[0.0, 0.0])
    with np.load(path) as archive:
        frame = load_discovery_frame(archive)
    assert np.allclose(frame.channels, [[1.0, 2.0], [2.0, 4.0]])
    assert np.allclose(frame.coefficients, [0.5, 1.0])
