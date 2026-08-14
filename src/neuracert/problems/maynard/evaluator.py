"""Programmatic access to established Maynard evaluators.

Imports remain lazy so importing :mod:`neuracert` does not initialize PyTorch,
SciPy or FLINT.
"""

from __future__ import annotations


def ratio_module():
    from maynard_tools.discovery import ratio

    return ratio


def polynomial_module(*, gated: bool = False):
    if gated:
        from maynard_tools.discovery import gated

        return gated
    from maynard_tools.discovery import factored

    return factored

