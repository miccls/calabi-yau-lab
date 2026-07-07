from __future__ import annotations

import numpy as np


def power_sums(r: np.ndarray, max_degree: int) -> np.ndarray:
    return np.stack([np.sum(r**k, axis=1) for k in range(1, max_degree + 1)], axis=1)


def elementary_symmetric(r: np.ndarray) -> np.ndarray:
    n, d = r.shape
    out = np.zeros((n, d + 1), dtype=float)
    out[:, 0] = 1.0
    for j in range(d):
        out[:, 1 : j + 2] += out[:, : j + 1] * r[:, [j]]
    return out[:, 1:]
