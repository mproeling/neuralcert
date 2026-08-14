"""Generic framework code cannot depend on the legacy Maynard implementation."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).parents[1] / "src" / "neuracert"


def test_only_maynard_plugin_imports_legacy_package() -> None:
    violations: list[str] = []
    for path in ROOT.rglob("*.py"):
        if "problems/maynard" in path.as_posix():
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name.startswith("maynard_tools") for name in names):
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not violations, violations

