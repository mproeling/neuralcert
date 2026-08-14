"""Enforce the trust boundary between discovery and certification."""

from __future__ import annotations

import ast
from pathlib import Path


DISCOVERY = Path(__file__).parents[1] / "src" / "maynard_tools" / "discovery"


def test_discovery_does_not_import_certification() -> None:
    violations: list[str] = []
    for path in DISCOVERY.glob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any("certification" in name for name in names):
                violations.append(f"{path.name}:{node.lineno}")
    assert not violations, "discovery imports certification at " + ", ".join(violations)

