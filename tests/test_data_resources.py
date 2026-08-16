"""Tests for datasets distributed inside the installable package."""

from __future__ import annotations

import csv

import pytest

from neuracert.data import DATASETS, dataset


def test_maynard_sweep_datasets_are_available() -> None:
    assert set(DATASETS) == {
        "maynard_R_sweep_eta",
        "maynard_geometric_eta_sweep",
    }
    for name in DATASETS:
        resource = dataset(name)
        with resource.open("r", encoding="utf-8", newline="") as stream:
            header = next(csv.reader(stream))
        assert "k" in header
        assert "cert" in header


def test_dataset_accepts_filename_and_rejects_unknown_names() -> None:
    assert dataset("maynard_R_sweep_eta.csv").name == "maynard_R_sweep_eta.csv"
    with pytest.raises(KeyError, match="unknown NeuraCert dataset"):
        dataset("missing")
