from __future__ import annotations

import numpy as np


def normalize_projective(z: np.ndarray) -> np.ndarray:
    """Normalize homogeneous points to unit Euclidean norm."""
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Cannot normalize a zero homogeneous coordinate vector")
    return z / norms


def fs_potential(z: np.ndarray) -> np.ndarray:
    """Fubini-Study potential log(sum |z_i|^2) in the chosen lift."""
    return np.log(np.sum(np.abs(z) ** 2, axis=1))


def normalized_radii(z: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    denom = np.sum(np.abs(z) ** 2, axis=1, keepdims=True)
    return np.abs(z) ** 2 / np.maximum(denom, eps)
