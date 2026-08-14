"""Tests for the unified certification-method interface."""

from __future__ import annotations

from maynard_tools.certification.cli import METHODS, _dispatch_parser


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
