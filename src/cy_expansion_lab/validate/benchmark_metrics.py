from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    formula: str
    unit: str
    aggregation: str
    convention: str
    literature_note: str

    def as_dict(self) -> dict[str, str]:
        return {
            "metric": self.name,
            "formula": self.formula,
            "unit": self.unit,
            "aggregation": self.aggregation,
            "convention": self.convention,
            "literature_note": self.literature_note,
        }


METRIC_DEFINITIONS = [
    MetricDefinition(
        name="centered_log_ma_rmse",
        formula="sqrt(mean((log(det(g)) - log(|Omega|^2) - c)^2)), c = mean(log(det(g)) - log(|Omega|^2))",
        unit="dimensionless log-volume residual",
        aggregation="unweighted sample RMSE over valid positive local patches",
        convention="centered log Monge-Ampere residual; root of mean square",
        literature_note="Not identical to sigma loss or raw volume-ratio error unless the source uses this exact centered-log convention.",
    ),
    MetricDefinition(
        name="centered_log_ma_mae",
        formula="mean(abs(log(det(g)) - log(|Omega|^2) - c)), c = mean(log(det(g)) - log(|Omega|^2))",
        unit="dimensionless log-volume residual",
        aggregation="unweighted sample MAE over valid positive local patches",
        convention="centered log Monge-Ampere residual; mean absolute value",
        literature_note="Useful for robust reporting; not interchangeable with MSE/RMSE or sigma measures.",
    ),
    MetricDefinition(
        name="raw_volume_ratio_mae",
        formula="mean(abs(exp(log(det(g)) - log(|Omega|^2)) / kappa - 1)), kappa = mean(exp(log(det(g)) - log(|Omega|^2)))",
        unit="dimensionless relative volume-ratio error",
        aggregation="unweighted sample MAE over valid positive local patches",
        convention="sample-normalized volume ratio; scale kappa is fitted from the same evaluation set",
        literature_note="Closest in spirit to Monge-Ampere volume-ratio errors; exact comparability requires identical sampling and normalization.",
    ),
    MetricDefinition(
        name="raw_volume_ratio_rmse",
        formula="sqrt(mean((exp(log(det(g)) - log(|Omega|^2)) / kappa - 1)^2)), kappa = mean(exp(log(det(g)) - log(|Omega|^2)))",
        unit="dimensionless relative volume-ratio error",
        aggregation="unweighted sample RMSE over valid positive local patches",
        convention="sample-normalized volume ratio; root mean square relative error",
        literature_note="Do not compare against MAE or sigma values without converting conventions.",
    ),
    MetricDefinition(
        name="sigma_l1_volume_ratio",
        formula="mean(abs(1 - exp(log(det(g)) - log(|Omega|^2)) / kappa)), kappa = mean(exp(log(det(g)) - log(|Omega|^2)))",
        unit="dimensionless L1 relative volume-ratio error",
        aggregation="unweighted sample mean over valid positive local patches",
        convention="sample-average proxy for Donaldson/cymetric-style sigma measure",
        literature_note="A benchmark proxy for published sigma measures; true integral comparability requires matching the sampling measure and volume normalization.",
    ),
    MetricDefinition(
        name="positivity_violation_rate",
        formula="mean(lambda_min(g) <= tolerance)",
        unit="fraction of local patches",
        aggregation="unweighted sample mean over evaluated local patches",
        convention="tolerance defaults to 0 unless otherwise stated",
        literature_note="Only comparable when the local chart sampling and eigenvalue tolerance match.",
    ),
    MetricDefinition(
        name="min_eigenvalue_min",
        formula="min_i lambda_min(g_i)",
        unit="local coordinate metric units",
        aggregation="minimum over evaluated local patches",
        convention="coordinate-dependent positivity diagnostic",
        literature_note="Not a scalar accuracy measure; useful for detecting non-Kahler or non-positive candidates.",
    ),
    MetricDefinition(
        name="ricci_scalar_abs_mean",
        formula="mean(abs(trace(g^{-1} Ric(g))))",
        unit="inverse local metric units",
        aggregation="unweighted sample mean over patches where Ricci was evaluated",
        convention="Ric(g) = -partial partialbar log(det(g))",
        literature_note="Coordinate/scaling conventions must be matched before numerical comparison to published Ricci scalar tables.",
    ),
]


def metric_definition_rows() -> list[dict[str, str]]:
    return [definition.as_dict() for definition in METRIC_DEFINITIONS]


def summarize_log_ma(
    raw_log_ma: np.ndarray,
    min_eigenvalues: np.ndarray,
    positivity_tolerance: float = 0.0,
    ricci_scalars: np.ndarray | None = None,
) -> dict[str, float | int | None]:
    raw_log_ma = np.asarray(raw_log_ma, dtype=float)
    min_eigenvalues = np.asarray(min_eigenvalues, dtype=float)
    finite = np.isfinite(raw_log_ma) & np.isfinite(min_eigenvalues) & (min_eigenvalues > positivity_tolerance)
    out: dict[str, float | int | None] = {
        "evaluated_count": int(len(raw_log_ma)),
        "valid_positive_count": int(np.sum(finite)),
        "positivity_violation_rate": float(1.0 - np.mean(min_eigenvalues > positivity_tolerance)) if len(min_eigenvalues) else None,
        "min_eigenvalue_min": float(np.min(min_eigenvalues)) if len(min_eigenvalues) else None,
        "min_eigenvalue_mean": float(np.mean(min_eigenvalues)) if len(min_eigenvalues) else None,
    }
    if not np.any(finite):
        out.update(
            {
                "log_ma_center": None,
                "centered_log_ma_rmse": None,
                "centered_log_ma_mae": None,
                "raw_volume_ratio_center": None,
                "raw_volume_ratio_mae": None,
                "raw_volume_ratio_rmse": None,
                "sigma_l1_volume_ratio": None,
            }
        )
        return out
    raw = raw_log_ma[finite]
    center = float(np.mean(raw))
    centered = raw - center
    ratio = np.exp(np.clip(raw, -700.0, 700.0))
    kappa = float(np.mean(ratio))
    rel = ratio / kappa - 1.0
    out.update(
        {
            "log_ma_center": center,
            "centered_log_ma_rmse": float(np.sqrt(np.mean(centered**2))),
            "centered_log_ma_mae": float(np.mean(np.abs(centered))),
            "centered_log_ma_max_abs": float(np.max(np.abs(centered))),
            "raw_volume_ratio_center": kappa,
            "raw_volume_ratio_mae": float(np.mean(np.abs(rel))),
            "raw_volume_ratio_rmse": float(np.sqrt(np.mean(rel**2))),
            "sigma_l1_volume_ratio": float(np.mean(np.abs(1.0 - ratio / kappa))),
        }
    )
    if ricci_scalars is not None:
        ricci = np.asarray(ricci_scalars, dtype=float)
        ricci = ricci[np.isfinite(ricci)]
        out.update(
            {
                "ricci_scalar_count": int(len(ricci)),
                "ricci_scalar_abs_mean": float(np.mean(np.abs(ricci))) if len(ricci) else None,
                "ricci_scalar_rms": float(np.sqrt(np.mean(ricci**2))) if len(ricci) else None,
            }
        )
    return out
