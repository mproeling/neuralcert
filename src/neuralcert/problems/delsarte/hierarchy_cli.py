"""CLI for the higher-order Loyfer-Linial/CJJ Delsarte hierarchy."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .hierarchy import build_lp, dual_dataset, polish_and_certify, solve_lp


def run_hierarchy(n: int, d: int, *, r: int = 3, dps: int = 100,
                  bits: int = 180, even: bool | None = None,
                  output: Path | None = None, verbose: bool = True):
    """Solve, polish, exactly certify and optionally export one hierarchy LP."""
    started = time.time()
    lp = build_lp(n, d, r=r, even=even, verbose=verbose)
    value_float, phi, duals = solve_lp(lp, "Obj")
    polished, certificate = polish_and_certify(
        lp, duals, "Obj", dps=dps, bits=bits, verbose=verbose,
    )
    if verbose:
        print(f"  explicit float dual value: {value_float:.12f}")
        print(f"  {certificate.report()} (mu={float(certificate.mu):.3e}) "
              f"[{time.time() - started:.1f}s]")
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "format": "neuralcert-delsarte-hierarchy-4",
            "n": n,
            "d": d,
            "r": r,
            "even": lp.even,
            "lp_value_float_dual": value_float,
            "polish": {
                "method": polished.method,
                "dps": polished.dps,
                "support_size": len(polished.support),
                "tight_columns": list(polished.tight_columns),
                "max_scaled_residual": float(polished.max_scaled_residual),
                "objective_value": str(polished.objective_value),
            },
            "certified_value": [int(certificate.value.numerator),
                                int(certificate.value.denominator)],
            "certified_float": float(certificate.value),
            "repair_mu": [int(certificate.mu.numerator),
                           int(certificate.mu.denominator)],
            "bound_on_A_lin": int(certificate.bound_on_A_lin()),
            "power_of_two_bound": int(certificate.power_of_two_bound()),
            "records": dual_dataset(lp, polished.multipliers, phi,
                                    certificate=certificate),
        }
        output.write_text(json.dumps(document) + "\n", encoding="utf-8")
        if verbose:
            print(f"  dataset -> {output}")
    return certificate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neuralcert-delsarte-hierarchy",
        description="Solve and exactly certify a higher-order binary Delsarte LP.",
    )
    parser.add_argument("n", type=int)
    parser.add_argument("d", type=int)
    parser.add_argument("--r", type=int, default=3, choices=range(1, 5))
    parser.add_argument("--dps", type=int, default=100)
    parser.add_argument("--bits", type=int, default=180)
    parity = parser.add_mutually_exclusive_group()
    parity.add_argument("--even", dest="even", action="store_true")
    parity.add_argument("--no-even", dest="even", action="store_false")
    parser.set_defaults(even=None)
    parser.add_argument("--output", type=Path,
                        help="JSON dataset path (default: hierarchy_dual_rR_N_D.json)")
    parser.add_argument("--no-output", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = None if args.no_output else (
        args.output or Path(f"hierarchy_dual_r{args.r}_{args.n}_{args.d}.json")
    )
    run_hierarchy(args.n, args.d, r=args.r, dps=args.dps, bits=args.bits,
                  even=args.even, output=output, verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
