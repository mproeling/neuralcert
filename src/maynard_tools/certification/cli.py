"""Unified command-line dispatcher for certification methods.

``poly`` retains the existing exact Karatsuba certifier.  ``ratio`` selects
the Arb directed-rounding certifier for a single power-1 ratio channel.  The
method-specific implementations continue to own their complete argument sets.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path


METHODS = ("poly", "ratio")


def _dispatch_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--method",
        choices=METHODS,
        default="poly",
        help="certification family: poly (default) or ratio",
    )
    return parser


def _method_was_explicit(argv: Sequence[str]) -> bool:
    return any(arg == "--method" or arg.startswith("--method=") for arg in argv)


def _npz_path(argv: Sequence[str]) -> Path | None:
    """Return a supplied NPZ path without consuming backend arguments."""
    for index, argument in enumerate(argv):
        if argument == "--npz" and index + 1 < len(argv):
            return Path(argv[index + 1])
        if argument.startswith("--npz="):
            return Path(argument.split("=", 1)[1])
    for argument in argv:
        if not argument.startswith("-") and argument.lower().endswith(".npz"):
            return Path(argument)
    return None


def _detect_npz_method(path: Path) -> str:
    """Identify a discovery schema from ZIP member names only."""
    import numpy as np

    try:
        with np.load(path) as archive:
            names = set(archive.files)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot inspect NumPy NPZ discovery export {path}: {exc}") from exc
    ratio = {"canonical", "sha256", "c_num", "c_den", "power"}
    poly = {"x_fine", "g_fine", "c", "R"}
    if ratio <= names:
        return "ratio"
    if poly <= names:
        return "poly"
    raise ValueError(
        f"unrecognised discovery NPZ schema in {path}; fields: {sorted(names)}")


def _print_overview() -> None:
    print(
        "usage: maynard-certify [--method {poly,ratio}] [method options]\n\n"
        "Certification methods:\n"
        "  poly   Existing exact polynomial/Karatsuba certifier (default)\n"
        "  ratio  Arb directed-rounding certifier for one power-1 channel\n\n"
        "Use 'maynard-certify --method METHOD --help' for method-specific "
        "options."
    )


def main(argv: Sequence[str] | None = None) -> int | None:
    """Run the selected certification backend without altering its arguments."""
    forwarded = list(sys.argv[1:] if argv is None else argv)
    explicit_method = _method_was_explicit(forwarded)
    options, method_argv = _dispatch_parser().parse_known_args(forwarded)

    if not explicit_method:
        path = _npz_path(method_argv)
        if path is not None and path.exists():
            options.method = _detect_npz_method(path)

    if any(arg in ("-h", "--help") for arg in method_argv) and not explicit_method:
        _print_overview()
        return 0

    if options.method == "poly":
        from . import karatsuba as implementation
    else:
        from . import ratio as implementation

    old_argv = sys.argv
    sys.argv = [old_argv[0], *method_argv]
    try:
        return implementation.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())
