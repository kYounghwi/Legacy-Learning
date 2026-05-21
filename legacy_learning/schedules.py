from __future__ import annotations

import math


def cosine_beta(step: int, max_steps: int) -> float:
    if max_steps <= 1:
        return 1.0
    lam = min(max(step / float(max_steps), 0.0), 1.0)
    return (1.0 - math.cos(math.pi * lam)) / 2.0


def get_beta(step: int, max_steps: int, schedule: str = "cosine", fixed_beta: float | None = None) -> float:
    if fixed_beta is not None:
        return float(fixed_beta)
    if schedule == "cosine":
        return cosine_beta(step, max_steps)
    if schedule == "linear":
        return min(max(step / float(max(max_steps, 1)), 0.0), 1.0)
    raise ValueError(f"Unknown beta schedule: {schedule}")
