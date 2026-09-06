"""Package-level imports must not activate optional certification backends."""


def test_top_level_packages_import() -> None:
    import maynard_tools
    import maynard_tools.certification
    import maynard_tools.discovery
    import maynard_tools.verifier

    import neuralcert

    assert maynard_tools.__version__ == "0.11.1"
    assert neuralcert.__version__ == "0.11.1"


def test_python_flint_is_a_required_dependency() -> None:
    import tomllib
    from pathlib import Path

    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    project = tomllib.loads(pyproject.read_text())["project"]
    assert "python-flint>=0.6" in project["dependencies"]
    assert "certification" not in project.get("optional-dependencies", {})
