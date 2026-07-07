from __future__ import annotations

import numpy as np


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float = 1e-6) -> np.ndarray:
    xtx = x.T @ x
    return np.linalg.solve(xtx + alpha * np.eye(x.shape[1]), x.T @ y)
