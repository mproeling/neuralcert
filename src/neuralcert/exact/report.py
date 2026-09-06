"""Turning exact rationals into human-readable CERTIFIED statements.

The one rule: a decimal shown next to the word "certified" must itself be a
valid bound.  Rounding to nearest is therefore wrong for a lower bound --
it can round UP past the true value and turn a theorem into a false claim
by one ulp.  `floor_decimal` rounds toward -infinity, `ceil_decimal` toward
+infinity, and the caller picks the one that weakens the bound.
"""

from __future__ import annotations

from fractions import Fraction


def floor_decimal(value: Fraction, digits: int) -> str:
    """Largest d-digit decimal <= value.  Safe for LOWER bounds."""
    value = Fraction(value)
    scale = 10 ** digits
    n = (value.numerator * scale) // value.denominator      # floor, signed
    sign = "-" if n < 0 else ""
    n = abs(n)
    whole, frac = divmod(n, scale)
    return f"{sign}{whole}.{frac:0{digits}d}" if digits else f"{sign}{whole}"


def ceil_decimal(value: Fraction, digits: int) -> str:
    """Smallest d-digit decimal >= value.  Safe for UPPER bounds."""
    value = Fraction(value)
    scale = 10 ** digits
    n = -((-value.numerator * scale) // value.denominator)
    sign = "-" if n < 0 else ""
    n = abs(n)
    whole, frac = divmod(n, scale)
    return f"{sign}{whole}.{frac:0{digits}d}" if digits else f"{sign}{whole}"


def write_certificate(path: str, *, problem: str, fingerprint: str,
                      exact_values: dict[str, Fraction],
                      bound_name: str, bound_value: Fraction,
                      direction: str = "lower", digits: int = 30,
                      metadata: dict | None = None,
                      integers: dict[str, int] | None = None) -> str:
    """Write a self-describing certificate file and return the decimal.

    `direction` is 'lower' or 'upper' and selects the safe rounding.  The
    file records every exact rational in full, so an independent verifier
    can re-derive the decimal without trusting this code.
    """
    if direction not in ("lower", "upper"):
        raise ValueError("direction must be 'lower' or 'upper'")
    dec = (floor_decimal(bound_value, digits) if direction == "lower"
           else ceil_decimal(bound_value, digits))
    lines = [
        f"# certkit certificate v1",
        f"problem      {problem}",
        f"fingerprint  {fingerprint}",
        f"direction    {direction}",
        f"bound_name   {bound_name}",
        f"bound_num    {bound_value.numerator}",
        f"bound_den    {bound_value.denominator}",
        f"bound_dec    {dec}",
    ]
    for key, val in (metadata or {}).items():
        lines.append(f"meta.{key}  {val}")
    for key, val in exact_values.items():
        lines.append(f"value.{key}.num  {Fraction(val).numerator}")
        lines.append(f"value.{key}.den  {Fraction(val).denominator}")
    for key, val in (integers or {}).items():
        lines.append(f"int.{key}  {val}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return dec
