from __future__ import annotations

import numpy as np


def hermitian_eigenvalue_stats(metrics: np.ndarray, tol: float = 1e-8) -> dict[str, float]:
    """Summarize positivity of a batch of Hermitian metrics."""
    if metrics.ndim != 3 or metrics.shape[1] != metrics.shape[2]:
        raise ValueError("Expected metric batch with shape (n, d, d)")
    eigvals = np.linalg.eigvalsh(metrics)
    min_eigs = np.min(eigvals, axis=1)
    return {
        "count": int(metrics.shape[0]),
        "min_eigenvalue": float(np.min(min_eigs)),
        "median_min_eigenvalue": float(np.median(min_eigs)),
        "mean_min_eigenvalue": float(np.mean(min_eigs)),
        "positivity_violation_rate": float(np.mean(min_eigs <= tol)),
    }


def log_determinants(metrics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return log determinants and a mask for positive-definite metrics."""
    eigvals = np.linalg.eigvalsh(metrics)
    positive = np.all(eigvals > 0.0, axis=1)
    logs = np.full(metrics.shape[0], np.nan, dtype=float)
    logs[positive] = np.sum(np.log(eigvals[positive]), axis=1)
    return logs, positive
