"""Unified command-line dispatcher for discovery methods.

``poly`` is the neural polynomial/channel search that was historically exposed
directly as ``maynard-discover``.  ``ratio`` is the clustered confluent-ratio
basis.  The dispatcher removes only its own ``--method`` option and leaves the
remaining arguments untouched for the selected implementation.
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
        help="discovery family: poly (default) or ratio",
    )
    return parser


def _method_was_explicit(argv: Sequence[str]) -> bool:
    return any(arg == "--method" or arg.startswith("--method=") for arg in argv)


def _print_overview() -> None:
    print(
        "usage: maynard-discover [--method {poly,ratio}] [method options]\n\n"
        "Discovery methods:\n"
        "  poly   Neural separable-channel discovery (default)\n"
        "  ratio  Clustered confluent-ratio basis discovery\n\n"
        "Use 'maynard-discover --method METHOD --help' for method-specific "
        "options."
    )


def main(argv: Sequence[str] | None = None) -> int | None:
    """Run the requested discovery implementation.

    ``argv`` is primarily useful for embedding and tests.  Concrete discovery
    modules retain their original parsers, so their complete CLI contracts stay
    available without duplicating hundreds of options in this dispatcher.
    """
    forwarded = list(sys.argv[1:] if argv is None else argv)
    explicit_method = _method_was_explicit(forwarded)
    options, method_argv = _dispatch_parser().parse_known_args(forwarded)

    if any(arg in ("-h", "--help") for arg in method_argv) and not explicit_method:
        _print_overview()
        return 0

    if options.method == "poly":
        from . import factored as implementation
    else:
        from . import ratio as implementation

    # Both imported historical implementations parse sys.argv themselves.
    # Restore it even when argparse exits or the numerical solver raises.
    old_argv = sys.argv
    sys.argv = [old_argv[0], *method_argv]
    try:
        return implementation.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())

