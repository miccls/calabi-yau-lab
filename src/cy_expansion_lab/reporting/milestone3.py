from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRIC_DEFINITIONS = [
    {
        "metric": "centered_log_ma_rmse",
        "formula": "sqrt(mean((log det g - log |Omega|^2 - mean)^2))",
        "unit": "dimensionless log residual",
        "aggregation": "RMSE over valid positive local patches",
        "comparable_to_literature": "No direct comparison unless literature uses the same centered log-Monge-Ampere convention.",
    },
    {
        "metric": "centered_log_ma_mae",
        "formula": "mean(abs(log det g - log |Omega|^2 - mean))",
        "unit": "dimensionless log residual",
        "aggregation": "MAE over valid positive local patches",
        "comparable_to_literature": "No direct comparison to MSE, sigma-loss, Ricci scalar, or raw volume-ratio error.",
    },
    {
        "metric": "positivity_violation_rate",
        "formula": "mean(lambda_min(g) <= tolerance)",
        "unit": "fraction of sampled local patches",
        "aggregation": "mean over local patches",
        "comparable_to_literature": "Only comparable if the same local patch sampling and tolerance are used.",
    },
    {
        "metric": "min_eigenvalue",
        "formula": "min eigenvalue of Hermitian local metric matrix",
        "unit": "local coordinate metric units",
        "aggregation": "minimum over sampled local patches",
        "comparable_to_literature": "Not directly comparable as an accuracy score; coordinate-dependent and useful as a positivity check.",
    },
]


LITERATURE_COMPARISONS = [
    {
        "source": "CYJAX arXiv:2211.12520",
        "method": "JAX algebraic/spectral Kahler-potential ansatz",
        "reported_metric": "paper/package accuracy metrics, not imported here",
        "unit_or_convention": "not directly normalized to this repo's centered log-MA RMSE",
        "numeric_value": "not copied",
        "comparison_note": "Primary relevance is architecture: algebraic ansatz preserves Kahlerity and patch compatibility.",
        "url": "https://arxiv.org/abs/2211.12520",
    },
    {
        "source": "Fundamental-domain projections arXiv:2407.06914",
        "method": "cymetric phi-model with non-trainable invariant canonicalization",
        "reported_metric": "cymetric-style losses/sigma measures in paper tables",
        "unit_or_convention": "not directly normalized to centered log-MA RMSE",
        "numeric_value": "not copied",
        "comparison_note": "Primary relevance is invariantization; projection layers can change smoothness behavior.",
        "url": "https://arxiv.org/abs/2407.06914",
    },
    {
        "source": "Invariant/symbolic models arXiv:2412.19778",
        "method": "extrinsic-symmetry-aware neural and symbolic approximations",
        "reported_metric": "Ricci curvature/scalar and model-specific losses",
        "unit_or_convention": "not directly normalized to centered log-MA RMSE",
        "numeric_value": "not copied",
        "comparison_note": "Primary relevance is symmetry-aware hypothesis space and symbolic distillation.",
        "url": "https://arxiv.org/abs/2412.19778",
    },
    {
        "source": "Sharp Edges arXiv:2606.26892",
        "method": "symmetry-aware/GNN Calabi-Yau metric pipeline",
        "reported_metric": "paper-specific Ricci/volume diagnostics",
        "unit_or_convention": "not directly normalized to centered log-MA RMSE",
        "numeric_value": "not copied",
        "comparison_note": "Primary relevance is careful handling of symmetry, sampling, and failure modes.",
        "url": "https://arxiv.org/abs/2606.26892",
    },
]


def write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str | Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot_training_curves(path: str | Path, history: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    if history:
        steps = [row["step"] for row in history]
        ax.plot(steps, [row["train_objective"] for row in history], label="train objective")
        ax.plot(steps, [row["validation_ma_rmse"] for row in history], label="validation log-MA RMSE")
    ax.set_xlabel("optimizer callback step")
    ax.set_ylabel("dimensionless residual")
    ax.set_title("Training measures")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_min_eigen_histogram(path: str | Path, records_by_model: dict[str, list[dict]]) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for name, records in records_by_model.items():
        vals = [r["min_eigenvalue"] for r in records if "min_eigenvalue" in r]
        ax.hist(vals, bins=24, alpha=0.45, label=name)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_xlabel("minimum local metric eigenvalue")
    ax.set_ylabel("patch count")
    ax.set_title("Metric positivity distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_residual_histogram(path: str | Path, records_by_model: dict[str, list[dict]]) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for name, records in records_by_model.items():
        vals = [abs(r["centered_log_ma_residual"]) for r in records if r.get("centered_log_ma_residual") is not None]
        ax.hist(vals, bins=24, alpha=0.45, label=name)
    ax.set_xlabel("|centered log-Monge-Ampere residual|")
    ax.set_ylabel("patch count")
    ax.set_title("Residual distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_stratum_residuals(path: str | Path, records_by_model: dict[str, list[dict]]) -> None:
    strata = sorted({r["stratum"] for records in records_by_model.values() for r in records if "stratum" in r})
    x = np.arange(len(strata))
    width = 0.8 / max(1, len(records_by_model))
    fig, ax = plt.subplots(figsize=(max(7.0, len(strata) * 0.55), 4.2))
    for offset, (name, records) in enumerate(records_by_model.items()):
        means = []
        for stratum in strata:
            vals = [
                abs(r["centered_log_ma_residual"])
                for r in records
                if r.get("stratum") == stratum and r.get("centered_log_ma_residual") is not None
            ]
            means.append(float(np.mean(vals)) if vals else np.nan)
        ax.bar(x + (offset - len(records_by_model) / 2) * width + width / 2, means, width=width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(strata, rotation=45, ha="right")
    ax.set_ylabel("mean |centered log-MA residual|")
    ax.set_title("Stratum-wise residuals")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_correction_vs_invariant(path: str | Path, p2: np.ndarray, correction: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.scatter(p2, correction, s=12, alpha=0.65)
    ax.set_xlabel("p2 = sum_i r_i^2")
    ax.set_ylabel("learned correction K - K_FS")
    ax.set_title("Correction versus invariant coordinate")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_comparison_bar(path: str | Path, rows: list[dict]) -> None:
    finite = [row for row in rows if row.get("centered_log_ma_rmse") is not None]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    names = [row["model"] for row in finite]
    values = [row["centered_log_ma_rmse"] for row in finite]
    ax.bar(names, values)
    ax.set_ylabel("centered log-MA RMSE")
    ax.set_title("Baseline and tuned-model comparison")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
