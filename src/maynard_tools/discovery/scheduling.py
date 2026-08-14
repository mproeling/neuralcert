"""Small scheduling and closed-form discovery helpers."""
from __future__ import annotations
import math


def make_warmup_cosine(warmup_iters: int, total_iters: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_iters:
            return (step + 1) / max(warmup_iters, 1)
        prog = (step - warmup_iters) / max(total_iters - warmup_iters, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))
    return lr_lambda


def split_iters_cost_balanced(total: int, ms: list[int], min_per_stage: int = 50):
    w = [1.0 / (m * m) for m in ms]
    s = sum(w)
    return [max(min_per_stage, round(total * wi / s)) for wi in w]


def closed_form_R(k: int, eps: float) -> float:
    """R for g == 1, factorial-free.
    A = L^k/k!,  B = [L^2 U^{k-1}/(k-1) - 2L U^k/k + U^{k+1}/(k+1)]/(k-2)!;
    cancel k!/(k-2)! = k(k-1) and factor through r = U/L:
        R = k^2 (k-1) L r^{k-1} [1/(k-1) - 2r/k + r^2/(k+1)],
    which is exactly 2k/(k+1) at eps = 0."""
    if abs(eps) < 1e-15:
        return 2.0 * k / (k + 1)
    L, U = 1.0 + eps, 1.0 - eps
    if L <= 0.0 or U <= 0.0:
        return float("nan")
    r = U / L
    bracket = 1.0 / (k - 1) - 2.0 * r / k + r * r / (k + 1)
    return k * k * (k - 1) * L * math.exp((k - 1) * math.log(r)) * bracket


def parse_int_list(s: str, name: str) -> list[int]:
    try:
        vals = [int(v) for v in s.split(",") if v.strip()]
    except ValueError as exc:
        raise ValueError(f"--{name} must be a comma list of ints") from exc
    if not vals or any(v < 1 for v in vals):
        raise ValueError(f"--{name} entries must be positive")
    return vals
