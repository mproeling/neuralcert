"""Auditable standard-library polynomial primitives for verifiers."""

from __future__ import annotations

from typing import Sequence


def polynomial_multiply(a: Sequence[int], b: Sequence[int]) -> list[int]:
    if not a or not b:
        return []
    result = [0] * (len(a) + len(b) - 1)
    for i, left in enumerate(a):
        for j, right in enumerate(b):
            result[i + j] += left * right
    return result


def polynomial_power(coefficients: Sequence[int], exponent: int) -> list[int]:
    if exponent < 0:
        raise ValueError("polynomial exponent must be non-negative")
    result = [1]
    base = list(coefficients)
    power = exponent
    while power:
        if power & 1:
            result = polynomial_multiply(result, base)
        power >>= 1
        if power:
            base = polynomial_multiply(base, base)
    return result

