#!/usr/bin/env python3
"""Independent exact verifier for Laguerre sign-uncertainty certificates.

This module intentionally imports only the Python standard library. It
reconstructs the rational Laguerre polynomial, checks all declared double
contacts, and applies its own Sturm implementation on the certified ray.
"""

from __future__ import annotations

import argparse
import json
import math
from fractions import Fraction


def _fraction(value) -> Fraction:
    return Fraction(str(value))


def _trim(poly):
    poly = list(poly)
    while len(poly) > 1 and poly[-1] == 0:
        poly.pop()
    return poly


def _derivative(poly):
    return _trim([i * poly[i] for i in range(1, len(poly))] or [Fraction(0)])


def _evaluate(poly, x):
    value = Fraction(0)
    for coefficient in reversed(poly):
        value = value * x + coefficient
    return value


def _divide(poly, divisor):
    remainder = list(poly)
    quotient = [Fraction(0)] * max(1, len(poly) - len(divisor) + 1)
    while len(remainder) >= len(divisor) and any(remainder):
        remainder = _trim(remainder)
        if len(remainder) < len(divisor):
            break
        offset = len(remainder) - len(divisor)
        coefficient = remainder[-1] / divisor[-1]
        quotient[offset] = coefficient
        for i, value in enumerate(divisor):
            remainder[i + offset] -= coefficient * value
        remainder = _trim(remainder)
    return _trim(quotient), _trim(remainder)


def _laguerre_coefficients(order: int, dimension: int):
    alpha = Fraction(dimension - 2, 2)
    coefficients = []
    for j in range(order + 1):
        binomial = Fraction(1)
        for t in range(order - j):
            binomial *= (alpha + j + 1 + t) / Fraction(t + 1)
        coefficients.append(binomial * Fraction((-2) ** j, math.factorial(j)))
    return coefficients


def _primitive_integers(poly):
    denominator = 1
    for coefficient in poly:
        denominator = math.lcm(denominator, coefficient.denominator)
    values = [int(coefficient * denominator) for coefficient in poly]
    content = 0
    for value in values:
        content = math.gcd(content, abs(value))
    return [value // (content or 1) for value in values]


def _negative_pseudo_remainder(left, right):
    delta = len(left) - len(right)
    leading = abs(right[-1])
    remainder = [value * leading ** (delta + 1) for value in left]
    degree = len(right) - 1
    for offset in range(delta, -1, -1):
        if len(remainder) - 1 < degree + offset:
            continue
        quotient = remainder[degree + offset] // right[-1]
        for i in range(degree + 1):
            remainder[i + offset] -= quotient * right[i]
    while len(remainder) > 1 and remainder[-1] == 0:
        remainder.pop()
    remainder = [-value for value in remainder]
    content = 0
    for value in remainder:
        content = math.gcd(content, abs(value))
    return [value // (content or 1) for value in remainder]


def _sturm_chain(poly):
    chain = [_primitive_integers(poly), _primitive_integers(_derivative(poly))]
    while len(chain[-1]) > 1 or chain[-1][0] != 0:
        remainder = _negative_pseudo_remainder(chain[-2], chain[-1])
        if not any(remainder):
            break
        chain.append(remainder)
        if len(remainder) == 1:
            break
    return chain


def _variations(signs):
    signs = [sign for sign in signs if sign]
    return sum(left != right for left, right in zip(signs, signs[1:]))


def _roots_on_ray(poly, start):
    chain = _sturm_chain(poly)
    at_start = []
    at_infinity = []
    for item in chain:
        value = _evaluate(item, start)
        at_start.append(1 if value > 0 else -1 if value < 0 else 0)
        at_infinity.append(1 if item[-1] > 0 else -1 if item[-1] < 0 else 0)
    return _variations(at_start) - _variations(at_infinity)


def verify(document: dict, log=print) -> bool:
    """Verify a self-contained exact Laguerre certificate."""
    if document.get("verdict") != "CERTIFIED":
        log("REJECT: document does not claim CERTIFIED")
        return False
    dimension = int(document["d"])
    sign = int(document["s"])
    n_basis = int(document["n_basis"])
    if dimension < 1 or sign not in {-1, 1} or n_basis < 2:
        log("REJECT: invalid problem parameters")
        return False
    orders = [(0 if sign == 1 else 1) + 2 * i for i in range(n_basis)]
    coefficients = [_fraction(value) for value in document["coeffs_laguerre"]]
    if len(coefficients) != n_basis:
        log("REJECT: coefficient count does not match n_basis")
        return False
    polynomial = [Fraction(0)] * (max(orders) + 1)
    for multiplier, order in zip(coefficients, orders):
        for degree, value in enumerate(_laguerre_coefficients(order, dimension)):
            polynomial[degree] += multiplier * value
    polynomial = _trim(polynomial)
    if not any(polynomial) or polynomial[0] != 0:
        log("REJECT: polynomial is zero or f(0) != 0")
        return False
    remainder = polynomial
    for contact in map(_fraction, document["taus"]):
        if _evaluate(polynomial, contact) != 0 or _evaluate(_derivative(polynomial), contact) != 0:
            log(f"REJECT: {contact} is not an exact double contact")
            return False
        for _ in range(2):
            remainder, error = _divide(remainder, [-contact, Fraction(1)])
            if any(error):
                log("REJECT: exact double-root deflation failed")
                return False
    start = _fraction(document["u_q"])
    if start <= 0 or remainder[-1] <= 0 or _evaluate(remainder, start) <= 0:
        log("REJECT: residual polynomial lacks positive ray orientation")
        return False
    roots = _roots_on_ray(remainder, start)
    if roots != 0:
        log(f"REJECT: Sturm counts {roots} residual roots after u_q")
        return False
    log(f"VERIFIED: A_{'+' if sign == 1 else '-'}({dimension}) <= sqrt({start}/pi)")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certificate")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    try:
        with open(args.certificate, encoding="utf-8") as handle:
            document = json.load(handle)
        ok = verify(document, (lambda *_: None) if args.quiet else print)
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
        if not args.quiet:
            print(f"REJECT: malformed certificate: {exc}")
        return 2
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
