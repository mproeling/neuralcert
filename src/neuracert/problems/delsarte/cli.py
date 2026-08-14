"""Problem-specific command line interface for Delsarte code bounds."""

from __future__ import annotations

import argparse
from pathlib import Path

from .problem import DelsarteCodeProblem


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neuracert-delsarte",
        description="Discover and exactly certify Delsarte LP code bounds.",
    )
    sub = parser.add_subparsers(dest="scheme", required=True)
    hamming = sub.add_parser("hamming", help="q-ary Hamming scheme H(n,q)")
    hamming.add_argument("--n", type=int, required=True)
    hamming.add_argument("--distance", "-d", type=int, required=True)
    hamming.add_argument("--q", type=int, default=2)
    johnson = sub.add_parser("johnson", help="constant-weight Johnson scheme J(n,w)")
    johnson.add_argument("--n", type=int, required=True)
    johnson.add_argument("--distance", "-d", type=int, required=True)
    johnson.add_argument("--weight", "-w", type=int, required=True)
    for command in (hamming, johnson):
        command.add_argument("--spectral-degree", type=int)
        command.add_argument("--bits", type=int, default=50)
        command.add_argument("--max-bits", type=int, default=400)
        command.add_argument("--tolerance", type=float, default=1e-6)
        command.add_argument("--certificate", type=Path,
                             help="write the self-contained exact certificate as JSON")
        command.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    problem = DelsarteCodeProblem(
        n=args.n,
        min_distance=args.distance,
        scheme=args.scheme,
        q=getattr(args, "q", 2),
        weight=getattr(args, "weight", None),
        spectral_degree=args.spectral_degree,
    )
    certificate = problem.certify(
        bits=args.bits,
        max_bits=args.max_bits,
        tolerance=args.tolerance,
        verbose=not args.quiet,
    )
    if args.certificate is not None:
        args.certificate.parent.mkdir(parents=True, exist_ok=True)
        args.certificate.write_text(certificate.to_json() + "\n", encoding="utf-8")
    if not args.quiet:
        print(certificate.report())
        if args.certificate is not None:
            print(f"certificate: {args.certificate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
