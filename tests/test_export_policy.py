"""An explicit output path must never be disabled by a validation gate."""

from __future__ import annotations

from pathlib import Path


DISCOVERY = Path(__file__).parents[1] / "src" / "maynard_tools" / "discovery"


def test_force_export_flag_and_suppression_are_gone() -> None:
    for name in ("factored.py", "gated.py"):
        source = (DISCOVERY / name).read_text()
        assert "--force-export" not in source
        assert "force_export" not in source
        assert "args.export = None" not in source


def test_failed_gate_keeps_explicit_export() -> None:
    for name in ("factored.py", "gated.py"):
        source = (DISCOVERY / name).read_text()
        assert "exporting unconverged candidate" in source
