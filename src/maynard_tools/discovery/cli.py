"""Unified command-line dispatcher for discovery methods.

``poly`` uses symmetric-polynomial channels; ``ratio`` uses rational channels
whose cost is independent of k. The dispatcher removes only its own
``--method`` option and leaves all method-specific arguments untouched.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


METHODS = ("poly", "ratio")

POLY_DESCRIPTION = (
    "Polynomial channels: symmetric-polynomial trial functions optimised by "
    "Adam followed by L-BFGS, certified by exact multimodular (CRT) evaluation "
    "of the Gram forms. Strongest at small k, where the extremal function is "
    "genuinely high-dimensional; cost grows with degree and channel count."
)
RATIO_DESCRIPTION = (
    "Rational channels g(t) = 1/(c + (k-1)t): a one-parameter family evaluated "
    "analytically via its characteristic function and a single FFT, certified "
    "in Arb ball arithmetic. Cost independent of k, so it reaches k ~ 1e9, and "
    "it attains log k - 0.334 + o(1) asymptotically."
)


def _dispatch_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--method",
        choices=METHODS,
        default="poly",
        help="discovery family: poly (default) or ratio; see method help for details",
    )
    return parser


def _method_was_explicit(argv: Sequence[str]) -> bool:
    return any(arg == "--method" or arg.startswith("--method=") for arg in argv)


def _print_overview() -> None:
    print(
        "usage: neuralcert discover [--method {poly,ratio}] [method options]\n\n"
        "Discovery methods:\n"
        f"\n--method poly\n    {POLY_DESCRIPTION}\n"
        f"\n--method ratio\n    {RATIO_DESCRIPTION}\n\n"
        "Use 'neuralcert discover --method METHOD --help' for method-specific options."
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
