"""Public verifier CLI uses the same method vocabulary as the workflows."""

from __future__ import annotations

import sys

import pytest

from maynard_tools.verifier import certificate


def test_certificate_verifier_requires_method(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["maynard-verify", "--help"])
    with pytest.raises(SystemExit) as exc:
        certificate.main()
    assert exc.value.code == 0
    assert "--method {poly,ratio}" in capsys.readouterr().out

