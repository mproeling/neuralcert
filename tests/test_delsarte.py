"""Integration tests for the Delsarte problem plugin."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from neuracert.core.registry import problems
from neuracert.problems.delsarte import DelsarteCodeProblem, binary_code_bound
from neuracert.problems.delsarte.cli import main as delsarte_main
from neuracert.problems.delsarte.verifier import verify


def test_delsarte_is_registered() -> None:
    problem = problems.create("delsarte", n=3, min_distance=2)
    assert isinstance(problem, DelsarteCodeProblem)


def test_known_binary_bound_and_independent_verifier() -> None:
    certificate = binary_code_bound(3, 2, verbose=False)
    assert certificate.integer_bound == 4
    assert verify(json.loads(certificate.to_json()), lambda *_: None)


def test_cli_writes_a_verifiable_certificate(tmp_path: Path) -> None:
    output = tmp_path / "delsarte.json"
    assert delsarte_main([
        "hamming", "--n", "3", "--distance", "2",
        "--certificate", str(output), "--quiet",
    ]) == 0
    assert verify(json.loads(output.read_text()), lambda *_: None)


def test_independent_verifier_has_no_neuracert_imports() -> None:
    path = Path(__file__).parents[1] / "src" / "neuracert" / "problems" / "delsarte" / "verifier.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert not [name for name in imports if name.startswith("neuracert")]
