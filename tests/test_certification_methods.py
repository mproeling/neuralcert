"""Tests for the unified certification-method interface."""

from __future__ import annotations

import hashlib

import numpy as np

from maynard_tools.certification.cli import METHODS, _detect_npz_method, _dispatch_parser
from maynard_tools.certification import ratio


def test_certification_methods_are_public() -> None:
    assert METHODS == ("poly", "ratio")


def test_poly_is_the_backwards_compatible_default() -> None:
    options, forwarded = _dispatch_parser().parse_known_args(
        ["--npz", "candidate.npz"]
    )
    assert options.method == "poly"
    assert forwarded == ["--npz", "candidate.npz"]


def test_ratio_selection_preserves_positional_npz() -> None:
    options, forwarded = _dispatch_parser().parse_known_args(
        ["--method", "ratio", "candidate.npz", "--prec", "256"]
    )
    assert options.method == "ratio"
    assert forwarded == ["candidate.npz", "--prec", "256"]


def test_ratio_selection_preserves_npz_option() -> None:
    options, forwarded = _dispatch_parser().parse_known_args(
        ["--method", "ratio", "--npz", "candidate.npz"]
    )
    assert options.method == "ratio"
    assert forwarded == ["--npz", "candidate.npz"]


def test_ratio_main_accepts_discovery_npz_option(monkeypatch) -> None:
    seen = {}

    def fake_load(path):
        seen["path"] = path
        return {
            "canonical": "k=10|1/5^-1*1",
            "sha": "0" * 64,
            "k": 10,
            "epsilon": ratio.Fraction(0),
            "cs": [ratio.Fraction(1, 5)],
            "ws": [ratio.Fraction(1)],
            "powers": [1],
            "R_discovery": 2.0,
            "ceiling": 3.0,
        }

    monkeypatch.setattr(ratio, "load", fake_load)
    monkeypatch.setattr(
        ratio,
        "certify",
        lambda *_args, **_kwargs: (
            1.9,
            2.0,
            {"N": {"lower": 1.0, "upper": 1.0},
             "D": {"lower": 1.0, "upper": 1.0}},
        ),
    )
    assert ratio.main(["--npz", "candidate.npz"]) is None
    assert seen["path"] == "candidate.npz"


def test_ratio_loads_discovery_export_schema(tmp_path) -> None:
    canonical = "k=10|1/5^-1*1"
    path = tmp_path / "candidate.npz"
    np.savez(
        path,
        k=10,
        mu=np.array([1]),
        c_num=[1],
        c_den=[5],
        power=[1],
        w_num=[1],
        w_den=[1],
        R_discovery=2.0,
        ceiling=3.0,
        sha256=hashlib.sha256(canonical.encode()).hexdigest(),
        canonical=canonical,
    )
    loaded = ratio.load(path)
    assert loaded["k"] == 10
    assert loaded["epsilon"] == 0
    assert loaded["cs"] == [ratio.Fraction(1, 5)]
    assert loaded["powers"] == [1]
    assert _detect_npz_method(path) == "ratio"


def test_poly_discovery_schema_is_detected(tmp_path) -> None:
    path = tmp_path / "poly.npz"
    np.savez(path, k=10, R=2.0, c=[1.0], x_fine=[0.0, 1.0],
             g_fine=[[1.0], [0.0]])
    assert _detect_npz_method(path) == "poly"
