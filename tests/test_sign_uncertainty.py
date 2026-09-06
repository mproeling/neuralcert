"""Tests for the Cohn-Gonçalves sign-uncertainty plugin."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from neuralcert.core.registry import problems
from neuralcert.problems.sign_uncertainty import Family, SignUncertaintyProblem
from neuralcert.problems.sign_uncertainty.certify import certify
from neuralcert.problems.sign_uncertainty.laguerre_basis import lag_coeffs_frac, orders_for
from neuralcert.problems.sign_uncertainty.laguerre import compare_published
from neuralcert.problems.sign_uncertainty.verifier import verify


def test_gc_alias_and_problem_validation() -> None:
    problem = problems.create("gc", dimension=1, sign=1)
    assert isinstance(problem, SignUncertaintyProblem)
    assert problem.name == "sign_uncertainty"


def test_shared_exact_laguerre_basis() -> None:
    assert orders_for(4, 1) == [0, 2, 4, 6]
    assert orders_for(3, -1) == [1, 3, 5]
    assert lag_coeffs_frac(2, 2) == [1, -4, 2]


def test_published_comparison_respects_reported_precision() -> None:
    assert compare_published(0.572989678, 0.572990).startswith("MATCHES")
    assert "genuine improvement" in compare_published(0.572706698, 0.572990)
    assert compare_published(0.573000, 0.572990) == "above by 0.000010"


def test_gaussian_family_folds_reciprocal_widths() -> None:
    family = Family(d=4, s=1, scales=np.array([0.5, 1.0, 2.0]))
    assert family.scales.tolist() == [1.0, 2.0]


def test_exact_certifier_and_independent_verifier() -> None:
    contacts = [1.816562, 2.835718, 16.549814, 20.577419, 25.491782]
    certificate = certify(1, 1, contacts, r1_hint=1.0314388,
                          tau_digits=6, verbose=False)
    assert certificate["verdict"] == "CERTIFIED"
    assert verify(certificate, lambda *_: None)


def test_verifier_imports_only_standard_library() -> None:
    path = (Path(__file__).parents[1] / "src" / "neuralcert" / "problems" /
            "sign_uncertainty" / "verifier.py")
    tree = ast.parse(path.read_text(), filename=str(path))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert set(imports) <= {"__future__", "argparse", "json", "math", "fractions"}


def test_certification_does_not_import_discovery_modules() -> None:
    path = (Path(__file__).parents[1] / "src" / "neuralcert" / "problems" /
            "sign_uncertainty" / "certify.py")
    tree = ast.parse(path.read_text(), filename=str(path))
    modules = [node.module or "" for node in ast.walk(tree)
               if isinstance(node, ast.ImportFrom)]
    assert not any(name.endswith(("laguerre", "gaussian_mixture", "collocate", "hybrid"))
                   for name in modules)
