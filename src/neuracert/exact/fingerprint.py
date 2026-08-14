"""Deterministic SHA-256 fingerprinting of arbitrary exact payload trees.

Why not repr().  At high degree an exact payload contains integers with
millions of decimal digits.  str()/repr() is both needlessly expensive and,
on Python >= 3.11, outright rejected by the int_max_str_digits limit.  We
hash the SIGNED BINARY representation incrementally instead.

The encoding is injective on the accepted types (a length prefix precedes
every variable-length field and every container), so two distinct payloads
cannot collide by structural ambiguity -- only by SHA-256 collision.

Accepted node types: int, bool, str, bytes, None, Fraction, and finite
sequences / mappings thereof.  Anything else raises rather than being
silently coerced: a fingerprint that quietly ignores part of the payload is
worse than no fingerprint, because it validates a stale checkpoint.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from fractions import Fraction

_TAG_INT = b"i"
_TAG_STR = b"s"
_TAG_BYTES = b"b"
_TAG_NONE = b"0"
_TAG_SEQ = b"["
_TAG_MAP = b"{"
_TAG_FRAC = b"/"


class Fingerprint:
    """Incremental structural hash."""

    def __init__(self, domain: str = "certkit") -> None:
        self._h = hashlib.sha256()
        self._h.update(domain.encode("utf-8"))

    def update(self, obj) -> "Fingerprint":
        self._encode(obj)
        return self

    def hexdigest(self, chars: int | None = None) -> str:
        d = self._h.hexdigest()
        return d if chars is None else d[:chars]

    # -- encoding ---------------------------------------------------------
    def _raw(self, tag: bytes, payload: bytes) -> None:
        self._h.update(tag)
        self._h.update(len(payload).to_bytes(8, "big"))
        self._h.update(payload)

    def _int(self, v: int) -> None:
        v = int(v)
        neg = v < 0
        mag = -v if neg else v
        raw = mag.to_bytes(max(1, (mag.bit_length() + 7) // 8), "big",
                           signed=False)
        self._h.update(_TAG_INT)
        self._h.update(b"-" if neg else b"+")
        self._h.update(len(raw).to_bytes(8, "big"))
        self._h.update(raw)

    def _encode(self, obj) -> None:
        if obj is None:
            self._h.update(_TAG_NONE)
        elif isinstance(obj, bool):
            self._int(1 if obj else 0)
        elif isinstance(obj, int):
            self._int(obj)
        elif isinstance(obj, Fraction):
            self._h.update(_TAG_FRAC)
            self._int(obj.numerator)
            self._int(obj.denominator)
        elif isinstance(obj, str):
            self._raw(_TAG_STR, obj.encode("utf-8"))
        elif isinstance(obj, (bytes, bytearray)):
            self._raw(_TAG_BYTES, bytes(obj))
        elif isinstance(obj, Mapping):
            self._h.update(_TAG_MAP)
            items = list(obj.items())
            self._h.update(len(items).to_bytes(8, "big"))
            for key, val in items:          # insertion order is significant
                self._encode(key)
                self._encode(val)
        elif isinstance(obj, Sequence):
            self._h.update(_TAG_SEQ)
            self._h.update(len(obj).to_bytes(8, "big"))
            for item in obj:
                self._encode(item)
        elif hasattr(obj, "__dataclass_fields__"):
            self._h.update(_TAG_MAP)
            fields = list(obj.__dataclass_fields__)
            self._h.update(len(fields).to_bytes(8, "big"))
            for name in fields:
                self._encode(name)
                self._encode(getattr(obj, name))
        else:
            raise TypeError(
                f"fingerprint: unhashable payload node of type "
                f"{type(obj).__name__}; convert it to ints/strs/sequences "
                f"explicitly rather than letting it be skipped")


def fingerprint(obj, domain: str = "certkit", chars: int = 32) -> str:
    """One-shot structural fingerprint."""
    return Fingerprint(domain).update(obj).hexdigest(chars)
