from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str | Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot_metric_bars(path: str | Path, rows: list[dict], metric: str, ylabel: str, title: str) -> None:
    finite = [row for row in rows if row.get(metric) is not None and np.isfinite(float(row[metric]))]
    fig, ax = plt.subplots(figsize=(max(6.8, len(finite) * 1.1), 4.2))
    names = [f"{row['target']}\n{row['model']}" for row in finite]
    values = [float(row[metric]) for row in finite]
    ax.bar(names, values)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_residual_histograms(path: str | Path, records_by_model: dict[str, list[dict]]) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for name, records in records_by_model.items():
        vals = [
            abs(float(r["centered_log_ma_residual"]))
            for r in records
            if r.get("centered_log_ma_residual") is not None
        ]
        if vals:
            ax.hist(vals, bins=24, alpha=0.45, label=name)
    ax.set_xlabel("|centered log-Monge-Ampere residual|, dimensionless")
    ax.set_ylabel("patch count")
    ax.set_title("Autodiff residual distributions")
    ax.legend(fontsize=8)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_min_eigen_histograms(path: str | Path, records_by_model: dict[str, list[dict]]) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for name, records in records_by_model.items():
        vals = [float(r["min_eigenvalue"]) for r in records if r.get("min_eigenvalue") is not None]
        if vals:
            ax.hist(vals, bins=24, alpha=0.45, label=name)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_xlabel("minimum Hermitian metric eigenvalue, local-coordinate units")
    ax.set_ylabel("patch count")
    ax.set_title("Autodiff positivity diagnostics")
    ax.legend(fontsize=8)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_k3_quintic_baseline(path: str | Path, rows: list[dict]) -> None:
    baseline = [row for row in rows if row["model"].endswith("fubini_study")]
    plot_metric_bars(
        path,
        baseline,
        metric="centered_log_ma_rmse",
        ylabel="centered log-MA RMSE, dimensionless",
        title="Fubini-Study baseline: K3 versus quintic",
    )


def markdown_summary(rows: list[dict], theory_count: int) -> str:
    lines = [
        "## Milestone 4: Autodiff Benchmark Infrastructure",
        "",
        "This milestone adds a JAX autodiff geometry path and a metric suite that keeps training objectives separate from evaluation metrics.",
        "",
        "Metric conventions are now machine-readable and distinguish centered log-Monge-Ampere residuals, sample-normalized volume-ratio errors, sigma-style L1 volume-ratio errors, positivity diagnostics, and opt-in Ricci scalar diagnostics.",
        "",
        f"The literature-theory registry currently contains {theory_count} entries, including Mirjanic-Mishra Proposition 3.3 and Section 6 analytical properties of `phi`.",
        "",
        "| target | model | centered log-MA RMSE | sigma L1 volume ratio | positivity violation | valid patches |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {target} | {model} | {rmse} | {sigma} | {pos} | {valid} |".format(
                target=row["target"],
                model=row["model"],
                rmse=_fmt(row.get("centered_log_ma_rmse")),
                sigma=_fmt(row.get("sigma_l1_volume_ratio")),
                pos=_fmt(row.get("positivity_violation_rate")),
                valid=row.get("valid_positive_count", ""),
            )
        )
    lines.extend(
        [
            "",
            "Limitations: current sigma values are sample-average proxies, not proof of identical integration measure to any published table. Ricci scalar diagnostics are implemented as an opt-in autodiff path because nested Hessians are substantially more expensive than log-determinant metrics.",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(x):
        return "n/a"
    return f"{x:.6g}"
