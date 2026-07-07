from __future__ import annotations

import numpy as np

from cy_expansion_lab.cy.projective import normalized_radii
from cy_expansion_lab.invariants.ik_basis import ik_placeholder
from cy_expansion_lab.invariants.symmetric import elementary_symmetric, power_sums


def invariant_table(z: np.ndarray, power_max: int = 6) -> tuple[np.ndarray, list[str]]:
    r = normalized_radii(z)
    p = power_sums(r, power_max)
    e = elementary_symmetric(r)
    ik, ik_names = ik_placeholder(r, max_degree=min(power_max, 6))
    features = np.concatenate([p[:, 1:], e[:, 1:], ik], axis=1)
    names = (
        [f"p{k}" for k in range(2, power_max + 1)]
        + [f"e{k}" for k in range(2, r.shape[1] + 1)]
        + ik_names
    )
    return features, names


def quotient_collapse_error(features: np.ndarray, y: np.ndarray, k: int = 8) -> dict[str, float]:
    """Nearest-neighbor target variation in invariant space."""
    n = len(y)
    if n <= k + 1:
        raise ValueError("Need more samples than neighbors")
    scaled = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-12)
    distances = np.sum((scaled[:, None, :] - scaled[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(distances, np.inf)
    nn = np.argpartition(distances, kth=k, axis=1)[:, :k]
    diffs = np.abs(y[:, None] - y[nn])
    return {
        "mean_abs_neighbor_delta": float(np.mean(diffs)),
        "median_abs_neighbor_delta": float(np.median(diffs)),
        "max_abs_neighbor_delta": float(np.max(diffs)),
    }
