from __future__ import annotations

import numpy as np


def ik_placeholder(r: np.ndarray, max_degree: int = 5) -> tuple[np.ndarray, list[str]]:
    """Extensible placeholder for thesis-style I_k invariant bases.

    This first version uses centered power products that are invariant under
    projective scaling, Fermat phase rotations, and coordinate permutations.
    """
    centered = r - np.mean(r, axis=1, keepdims=True)
    feats = []
    names = []
    for k in range(2, max_degree + 1):
        feats.append(np.sum(centered**k, axis=1))
        names.append(f"I{k}_centered_power")
    return np.stack(feats, axis=1), names
