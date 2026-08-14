"""Registry behavior is part of the plugin API contract."""

from __future__ import annotations

import pytest

from neuracert.core.registry import Registry, problems


def test_registry_registers_and_constructs_plugins() -> None:
    registry = Registry("demo")
    registry.register("item", lambda value=1: {"value": value})
    assert registry.names() == ("item",)
    assert registry.create("ITEM", value=3) == {"value": 3}
    with pytest.raises(KeyError, match="already registered"):
        registry.register("item", object())


def test_maynard_problem_is_registered() -> None:
    import neuracert.problems  # noqa: F401

    problem = problems.create("maynard", k=51, method="ratio")
    assert problem.name == "maynard"
    assert problem.k == 51

