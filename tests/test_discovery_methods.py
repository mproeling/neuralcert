"""Tests for the unified discovery-method interface."""

from __future__ import annotations

import warnings

import pytest

from maynard_tools.discovery.cli import (
    METHODS,
    POLY_DESCRIPTION,
    RATIO_DESCRIPTION,
    _dispatch_parser,
)
from maynard_tools.discovery.factored import warn_large_k


def test_discovery_methods_are_public() -> None:
    assert METHODS == ("poly", "ratio")


def test_poly_is_the_backwards_compatible_default() -> None:
    options, forwarded = _dispatch_parser().parse_known_args(["--k", "51"])
    assert options.method == "poly"
    assert forwarded == ["--k", "51"]


def test_ratio_selection_is_not_forwarded() -> None:
    options, forwarded = _dispatch_parser().parse_known_args(
        ["--method", "ratio", "--k", "51", "--mu", "2,1"]
    )
    assert options.method == "ratio"
    assert forwarded == ["--k", "51", "--mu", "2,1"]


def test_method_descriptions_explain_the_tradeoff() -> None:
    assert "Adam followed by L-BFGS" in POLY_DESCRIPTION
    assert "exact multimodular (CRT)" in POLY_DESCRIPTION
    assert "single FFT" in RATIO_DESCRIPTION
    assert "k ~ 1e9" in RATIO_DESCRIPTION
    assert "log k - 0.334 + o(1)" in RATIO_DESCRIPTION


def test_factored_warns_above_k_500_and_points_to_ratio() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_large_k(500)
    assert caught == []

    with pytest.warns(RuntimeWarning, match=r"--method ratio"):
        warn_large_k(501)
