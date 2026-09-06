"""Minimal and convenience problem contracts stay intentionally small."""

from __future__ import annotations

from neuralcert.core.problem import FunctionalProblem, Problem

from test_generic_discovery import QuadraticProblem


def test_functional_problem_satisfies_runtime_protocol() -> None:
    problem = QuadraticProblem()
    assert isinstance(problem, FunctionalProblem)
    assert isinstance(problem, Problem)
