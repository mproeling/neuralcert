"""The verifier is colocated for convenience but computationally independent."""

from __future__ import annotations

import ast
from pathlib import Path


VERIFIER = Path(__file__).parents[1] / "src" / "maynard_tools" / "verifier"


def test_verifier_does_not_import_package_implementations() -> None:
    violations: list[str] = []
    forbidden = ("maynard_tools.discovery", "maynard_tools.certification")
    for path in VERIFIER.glob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name.startswith(forbidden) for name in names):
                violations.append(f"{path.name}:{node.lineno}")
    assert not violations, "verifier dependency violation at " + ", ".join(violations)


def test_verifier_contains_no_relative_imports() -> None:
    for path in VERIFIER.glob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        assert not any(
            isinstance(node, ast.ImportFrom) and node.level
            for node in ast.walk(tree)
        ), f"{path.name} contains a relative import"

