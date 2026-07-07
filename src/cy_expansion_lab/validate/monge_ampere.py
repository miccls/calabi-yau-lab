from __future__ import annotations

import numpy as np

from cy_expansion_lab.validate.kahler import log_determinants


def volume_ratio_proxy(pred: np.ndarray, log_omega_proxy: np.ndarray) -> dict[str, float]:
    """Proxy residual until full complex Hessian Monge-Ampere is implemented."""
    centered = pred - np.mean(pred)
    rhs = log_omega_proxy - np.mean(log_omega_proxy)
    residual = centered - rhs
    return {
        "volume_proxy_mae": float(np.mean(np.abs(residual))),
        "volume_proxy_rmse": float(np.sqrt(np.mean(residual**2))),
    }


def monge_ampere_residual(metrics: np.ndarray, omega_density: np.ndarray) -> dict[str, float]:
    """Compute centered local Monge-Ampere residuals.

    The additive constant is fixed by centering log det(g) - log |Omega|^2 over
    the valid positive-definite samples.
    """
    logdet, positive = log_determinants(metrics)
    omega_density = np.asarray(omega_density, dtype=float)
    finite = positive & np.isfinite(logdet) & np.isfinite(omega_density) & (omega_density > 0.0)
    if not np.any(finite):
        return {
            "valid_count": 0,
            "invalid_rate": 1.0,
            "ma_residual_mean_abs": None,
            "ma_residual_rmse": None,
            "ma_residual_max_abs": None,
            "ma_constant": None,
        }
    raw = logdet[finite] - np.log(omega_density[finite])
    constant = float(np.mean(raw))
    residual = raw - constant
    return {
        "valid_count": int(np.sum(finite)),
        "invalid_rate": float(1.0 - np.mean(finite)),
        "ma_residual_mean_abs": float(np.mean(np.abs(residual))),
        "ma_residual_rmse": float(np.sqrt(np.mean(residual**2))),
        "ma_residual_max_abs": float(np.max(np.abs(residual))),
        "ma_constant": constant,
    }
