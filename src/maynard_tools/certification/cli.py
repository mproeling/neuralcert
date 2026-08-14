"""Unified command-line dispatcher for certification methods.

``poly`` retains the existing exact Karatsuba certifier.  ``ratio`` selects
the Arb directed-rounding certifier for a single power-1 ratio channel.  The
method-specific implementations continue to own their complete argument sets.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


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

