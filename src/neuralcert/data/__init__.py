"""Bundled reference datasets for reproducible NeuralCert experiments."""

from __future__ import annotations

from importlib.resources import files
from importlib.resources.abc import Traversable


GEOMETRIC_ETA_SWEEP = "maynard_geometric_eta_sweep.csv"
R_SWEEP_ETA = "maynard_R_sweep_eta.csv"

DATASETS = {
    "maynard_geometric_eta_sweep": GEOMETRIC_ETA_SWEEP,
    "maynard_R_sweep_eta": R_SWEEP_ETA,
}


def dataset(name: str) -> Traversable:
    """Return an importlib resource for a bundled CSV dataset.

    ``name`` may be either a stable dataset key from :data:`DATASETS` or the
    exact CSV filename. The returned resource also works when NeuralCert is
    installed from a wheel or imported from a zip archive.
    """
    filename = DATASETS.get(name, name)
    if filename not in DATASETS.values():
        available = ", ".join(sorted(DATASETS))
        raise KeyError(f"unknown NeuralCert dataset {name!r}; available: {available}")
    return files(__package__).joinpath(filename)


__all__ = ["DATASETS", "GEOMETRIC_ETA_SWEEP", "R_SWEEP_ETA", "dataset"]
