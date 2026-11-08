"""Small statistics helpers shared by the exploration and verification scripts.

Keeping one implementation avoids the drift that appeared when each script
carried its own copy of the Benjamini-Hochberg step.
"""
from __future__ import annotations

import numpy as np


def bh(p_values) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, monotone and capped at 1."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return p.copy()
    order = np.argsort(p)
    adjusted = np.empty(n)
    running = 1.0
    for rank, index in enumerate(order[::-1]):
        k = n - rank
        running = min(running, p[index] * n / k)
        adjusted[index] = running
    return np.clip(adjusted, 0.0, 1.0)


def holm(p_values) -> np.ndarray:
    """Holm step-down adjusted p-values, monotone and capped at 1."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return p.copy()
    order = sorted(range(n), key=lambda i: p[i])
    adjusted = np.empty(n)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (n - rank) * p[index])
        adjusted[index] = min(1.0, running)
    return adjusted
