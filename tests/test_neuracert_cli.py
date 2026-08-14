"""Tests for the single public ``neuracert`` console command."""

from __future__ import annotations

import tomllib
from pathlib import Path

from neuracert import cli


def test_package_exposes_only_one_console_script() -> None:
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    document = tomllib.loads(pyproject.read_text())
    assert document["project"]["scripts"] == {"neuracert": "neuracert.cli:main"}


def test_discover_forwards_to_maynard_backend(monkeypatch) -> None:
    import maynard_tools.discovery.cli as discovery

    seen = {}
    monkeypatch.setattr(discovery, "main", lambda args: seen.setdefault("args", args))
    assert cli.main(["discover", "--method", "ratio", "--k", "51"]) == [
        "--method", "ratio", "--k", "51",
    ]
    assert seen["args"] == ["--method", "ratio", "--k", "51"]


def test_certify_forwards_npz_option(monkeypatch) -> None:
    import maynard_tools.certification.cli as certification

    seen = {}
    monkeypatch.setattr(certification, "main", lambda args: seen.setdefault("args", args))
    expected = ["--method", "ratio", "--npz", "candidate.npz"]
    assert cli.main(["certify", *expected]) == expected
    assert seen["args"] == expected


def test_problem_group_help_is_available(capsys) -> None:
    assert cli.main(["delsarte", "--help"]) == 0
    assert "delsarte {bound,hierarchy,verify}" in capsys.readouterr().out
    assert cli.main(["sign", "--help"]) == 0
    assert "sign {discover,collocate,laguerre,hybrid,certify,verify}" in capsys.readouterr().out


def test_top_level_help_lists_general_workflows(capsys) -> None:
    assert cli.main(["--help"]) == 0
    output = capsys.readouterr().out
    assert "discover" in output
    assert "certify" in output
    assert "delsarte" in output
    assert "sign" in output
