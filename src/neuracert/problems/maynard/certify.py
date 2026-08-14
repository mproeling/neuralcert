"""Lazy adapters for specialized Maynard certification backends."""

from __future__ import annotations


def ratio_module():
    from maynard_tools.certification import ratio

    return ratio


def polynomial_module():
    from maynard_tools.certification import karatsuba

    return karatsuba

