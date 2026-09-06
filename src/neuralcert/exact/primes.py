"""Deterministic 64-bit primality and prime-budget selection.

Lifted verbatim (modulo naming) from the Maynard CRT certifier; there is
nothing problem-specific here, so it moves into the core untouched.

Determinism note.  Miller-Rabin against the twelve prime bases 2..37 is
PROVEN for all n < 318665857834031151167461 ~ 3.19e23 (Sorenson & Webster,
Math. Comp. 86 (2017) 985-1003; OEIS A014233).  Every modulus generated here
is < 2^62 ~ 4.6e18, five orders of magnitude inside the proven range, so
`is_prime_u64` is a decision procedure and not a probabilistic test.
"""

from __future__ import annotations

_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)

DEFAULT_START = (1 << 62) - 1
_FLOOR = 1 << 61


def is_prime_u64(n: int) -> bool:
    """Deterministic primality for n < 3.19e23."""
    if n < 2:
        return False
    for q in _MR_BASES:
        if n % q == 0:
            return n == q
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in _MR_BASES:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def gen_primes(count: int, start: int = DEFAULT_START) -> list[int]:
    """`count` distinct primes descending from `start` (all > 2^61)."""
    out: list[int] = []
    cand = start | 1
    while len(out) < count:
        if is_prime_u64(cand):
            out.append(cand)
        cand -= 2
        if cand < _FLOOR:  # pragma: no cover
            raise RuntimeError("prime pool exhausted (impossible in practice)")
    return out


def gen_primes_for_bound(bound: int, extra: int = 0,
                         start: int = DEFAULT_START) -> tuple[list[int], int]:
    """Primes with prod > 4*bound, plus `extra` held-out verification primes.

    The centered-CRT requirement is prod > 2*|S|; we take 4*bound for a
    two-bit safety margin and, decisively, we test the ACTUAL product rather
    than a 61-bit-per-prime estimate.  Returns (primes, n_reconstruction):
    the first n_reconstruction primes are the ones the reconstruction uses,
    the tail is held out for independent re-checking.
    """
    if bound < 0:
        raise ValueError("bound must be non-negative")
    target = 4 * bound
    out: list[int] = []
    prod = 1
    cand = start | 1
    while prod <= target or not out:
        if is_prime_u64(cand):
            out.append(cand)
            prod *= cand
        cand -= 2
        if cand < _FLOOR:  # pragma: no cover
            raise RuntimeError("prime pool exhausted (impossible in practice)")
    n_rec = len(out)
    while len(out) < n_rec + extra:
        if is_prime_u64(cand):
            out.append(cand)
        cand -= 2
        if cand < _FLOOR:  # pragma: no cover
            raise RuntimeError("prime pool exhausted (impossible in practice)")
    return out, n_rec
