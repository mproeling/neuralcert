"""Single command-line entrypoint for every bundled NeuraCert problem."""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence


HELP = """usage: neuracert COMMAND [options]

Core workflows:
  discover              Maynard discovery (--method poly|ratio)
  certify               Maynard certification (--method poly|ratio)
  verify                Independent Maynard certificate verification
  verify-direct         Direct numerical Maynard I/J diagnostic

Delsarte workflows:
  delsarte bound         Hamming/Johnson LP bound and certificate
  delsarte hierarchy     Higher-order Loyfer-Linial/CJJ hierarchy
  delsarte verify        Independent Delsarte certificate verification

Sign-uncertainty workflows:
  sign discover          Gaussian-mixture structure discovery
  sign collocate         Gaussian-mixture collocation
  sign laguerre          Global Laguerre LP
  sign hybrid            Laguerre/Gaussian A/B diagnostic
  sign certify           Exact rational Sturm certification
  sign verify            Independent certificate verification

Specialized Maynard workflows:
  discover-gated
  certify-epsilon
  certify-flint
  certify-crt
  certify-crt-ball
  certify-crt-scaled

Run `neuracert COMMAND --help` or `neuracert sign|delsarte COMMAND --help`
for workflow-specific options.
"""


def _legacy_main(function: Callable, args: list[str]):
    """Call a legacy argparse main that reads ``sys.argv`` directly."""
    previous = sys.argv
    sys.argv = ["neuracert", *args]
    try:
        return function()
    finally:
        sys.argv = previous


def _delsarte(args: list[str]):
    if not args or args[0] in {"-h", "--help"}:
        print("usage: neuracert delsarte {bound,hierarchy,verify} [options]")
        return 0
    stage, rest = args[0], args[1:]
    if stage == "bound":
        from neuracert.problems.delsarte.cli import main
        return main(rest)
    if stage == "hierarchy":
        from neuracert.problems.delsarte.hierarchy_cli import main
        return main(rest)
    if stage == "verify":
        from neuracert.problems.delsarte.verifier import main
        return main(["neuracert", *rest])
    raise SystemExit(f"unknown Delsarte workflow {stage!r}")


def _sign(args: list[str]):
    if not args or args[0] in {"-h", "--help"}:
        print("usage: neuracert sign {discover,collocate,laguerre,hybrid,certify,verify} [options]")
        return 0
    stage, rest = args[0], args[1:]
    if stage == "verify":
        from neuracert.problems.sign_uncertainty.verifier import main
        return main(rest)
    modules = {
        "discover": "gaussian_mixture",
        "collocate": "collocate",
        "laguerre": "laguerre",
        "hybrid": "hybrid",
        "certify": "certify",
    }
    try:
        module_name = modules[stage]
    except KeyError as exc:
        raise SystemExit(f"unknown sign-uncertainty workflow {stage!r}") from exc
    module = __import__(
        f"neuracert.problems.sign_uncertainty.{module_name}",
        fromlist=["main"],
    )
    return _legacy_main(module.main, rest)


def main(argv: Sequence[str] | None = None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(HELP)
        return 0
    command, rest = args[0], args[1:]
    if command == "discover":
        module = __import__("maynard_tools.discovery.cli", fromlist=["main"])
        return module.main(rest)
    if command == "certify":
        module = __import__("maynard_tools.certification.cli", fromlist=["main"])
        return module.main(rest)
    if command == "verify":
        module = __import__("maynard_tools.verifier.certificate", fromlist=["main"])
        return _legacy_main(module.main, rest)
    if command == "verify-direct":
        module = __import__("maynard_tools.verifier.direct_ij", fromlist=["main"])
        return _legacy_main(module.main, rest)
    if command == "delsarte":
        return _delsarte(rest)
    if command == "sign":
        return _sign(rest)

    specialized = {
        "discover-gated": "maynard_tools.discovery.gated",
        "certify-epsilon": "maynard_tools.certification.epsilon_karatsuba",
        "certify-flint": "maynard_tools.certification.flint_streaming",
        "certify-crt": "maynard_tools.certification.crt",
        "certify-crt-ball": "maynard_tools.certification.crt_ball",
        "certify-crt-scaled": "maynard_tools.certification.crt_ball_scaled",
    }
    if command in specialized:
        module = __import__(specialized[command], fromlist=["main"])
        return _legacy_main(module.main, rest)
    raise SystemExit(f"unknown NeuraCert command {command!r}; run `neuracert --help`")


if __name__ == "__main__":
    raise SystemExit(main())
