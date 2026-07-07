from __future__ import annotations

import numpy as np


def orbit_consistency(y: np.ndarray, group_size: int = 3) -> dict[str, float]:
    if len(y) % group_size != 0:
        raise ValueError("Expected concatenated original/phase/permutation orbit blocks")
    blocks = y.reshape(group_size, len(y) // group_size)
    deltas = np.abs(blocks - blocks[0:1])
    return {
        "mean_orbit_abs_delta": float(np.mean(deltas)),
        "max_orbit_abs_delta": float(np.max(deltas)),
    }
