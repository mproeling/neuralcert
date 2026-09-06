"""Maynard candidate export adapters."""

from __future__ import annotations

from typing import Any


def export_ratio(*args: Any, **kwargs: Any):
    from maynard_tools.discovery.ratio import export

    return export(*args, **kwargs)

