"""Tests for the unified discovery-method interface."""

from __future__ import annotations

from maynard_tools.discovery.cli import METHODS, _dispatch_parser


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
