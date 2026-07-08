from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cy_expansion_lab.reporting.milestone4 import write_csv, write_json


def plot_best_training_curves(path: str | Path, histories: dict[str, list[dict]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for label, history in histories.items():
        if not history:
            continue
        steps = [row["step"] for row in history]
        axes[0].plot(steps, [row["validation_centered_log_ma_rmse"] for row in history], label=label)
        axes[1].plot(steps, [row["validation_sigma_l1_volume_ratio"] for row in history], label=label)
    axes[0].set_xlabel("L-BFGS callback step")
    axes[0].set_ylabel("validation centered log-MA RMSE")
    axes[0].set_title("Validation log-volume residual")
    axes[1].set_xlabel("L-BFGS callback step")
    axes[1].set_ylabel("validation sigma-style L1")
    axes[1].set_title("Validation volume-ratio proxy")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, path)


def plot_loss_components(path: str | Path, rows: list[dict]) -> None:
    finite = [row for row in rows if row.get("selected")]
    metrics = ["train_centered_log_ma_mse", "train_volume_ratio_mse", "train_sigma_l1_smooth", "train_positivity_penalty"]
    fig, ax = plt.subplots(figsize=(max(7.0, len(finite) * 1.1), 4.4))
    x = np.arange(len(finite))
    width = 0.8 / len(metrics)
    for i, metric in enumerate(metrics):
        ax.bar(x + (i - len(metrics) / 2) * width + width / 2, [_safe_float(row.get(metric)) for row in finite], width, label=metric)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{row['target_short']}\n{row['candidate']}" for row in finite], rotation=25, ha="right")
    ax.set_ylabel("training loss component")
    ax.set_title("Selected-model loss decomposition")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, path)


def plot_metric_bars(path: str | Path, rows: list[dict], metric: str, ylabel: str, title: str) -> None:
    finite = [row for row in rows if row.get(metric) is not None and np.isfinite(float(row[metric]))]
    fig, ax = plt.subplots(figsize=(max(7.0, len(finite) * 0.9), 4.4))
    ax.bar([f"{row['target_short']}\n{row['model']}" for row in finite], [float(row[metric]) for row in finite])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    _save(fig, path)


def plot_model_size_vs_accuracy(path: str | Path, rows: list[dict]) -> None:
    finite = [row for row in rows if row.get("validation_centered_log_ma_rmse") is not None]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for target in sorted({row["target_short"] for row in finite}):
        target_rows = [row for row in finite if row["target_short"] == target]
        ax.scatter(
            [int(row["basis_size"]) for row in target_rows],
            [float(row["validation_centered_log_ma_rmse"]) for row in target_rows],
            label=target,
            alpha=0.8,
        )
    ax.set_xlabel("basis size")
    ax.set_ylabel("validation centered log-MA RMSE")
    ax.set_title("Model size versus validation accuracy")
    ax.legend()
    fig.tight_layout()
    _save(fig, path)


def plot_ablation_bars(path: str | Path, rows: list[dict], group_key: str, metric: str, title: str) -> None:
    finite = [row for row in rows if row.get(metric) is not None]
    groups = sorted({str(row[group_key]) for row in finite})
    targets = sorted({row["target_short"] for row in finite})
    fig, ax = plt.subplots(figsize=(max(7.0, len(groups) * 0.9), 4.4))
    x = np.arange(len(groups))
    width = 0.8 / max(1, len(targets))
    for offset, target in enumerate(targets):
        vals = []
        for group in groups:
            subset = [row for row in finite if row["target_short"] == target and str(row[group_key]) == group]
            vals.append(float(np.min([row[metric] for row in subset])) if subset else np.nan)
        ax.bar(x + (offset - len(targets) / 2) * width + width / 2, vals, width=width, label=target)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=25, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    _save(fig, path)


def plot_records_histogram(path: str | Path, records_by_model: dict[str, list[dict]], field: str, xlabel: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for label, records in records_by_model.items():
        vals = []
        for record in records:
            value = record.get(field)
            if value is not None:
                vals.append(abs(float(value)) if "residual" in field else float(value))
        if vals:
            ax.hist(vals, bins=24, alpha=0.45, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("patch count")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, path)


def plot_stratum_residuals(path: str | Path, records_by_model: dict[str, list[dict]]) -> None:
    strata = sorted({str(r["stratum"]).split(":")[0] for records in records_by_model.values() for r in records if "stratum" in r})
    models = list(records_by_model)
    fig, ax = plt.subplots(figsize=(max(8.0, len(strata) * 0.8), 4.6))
    x = np.arange(len(strata))
    width = 0.8 / max(1, len(models))
    for offset, model in enumerate(models):
        vals = []
        for stratum in strata:
            residuals = [
                abs(float(r["centered_log_ma_residual"]))
                for r in records_by_model[model]
                if str(r.get("stratum", "")).split(":")[0] == stratum and r.get("centered_log_ma_residual") is not None
            ]
            vals.append(float(np.mean(residuals)) if residuals else np.nan)
        ax.bar(x + (offset - len(models) / 2) * width + width / 2, vals, width=width, label=model)
    ax.set_xticks(x)
    ax.set_xticklabels(strata, rotation=30, ha="right")
    ax.set_ylabel("mean |centered log-MA residual|")
    ax.set_title("Residuals by stratum")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, path)


def plot_residual_vs_p2(path: str | Path, records: list[dict], p2_by_index: dict[int, float], title: str) -> None:
    xs = []
    ys = []
    colors = []
    strata = sorted({str(r.get("stratum", "")).split(":")[0] for r in records})
    stratum_to_int = {name: i for i, name in enumerate(strata)}
    for record in records:
        if record.get("centered_log_ma_residual") is None:
            continue
        idx = int(record["index"])
        if idx not in p2_by_index:
            continue
        xs.append(float(p2_by_index[idx]))
        ys.append(abs(float(record["centered_log_ma_residual"])))
        colors.append(stratum_to_int[str(record.get("stratum", "")).split(":")[0]])
    fig, ax = plt.subplots(figsize=(6.3, 4.4))
    if xs:
        scatter = ax.scatter(xs, ys, c=colors, cmap="tab10", s=22, alpha=0.75)
        handles, _ = scatter.legend_elements()
        ax.legend(handles, strata, title="stratum", fontsize=7)
    ax.set_xlabel("p2 = sum_i r_i^2")
    ax.set_ylabel("|centered log-MA residual|")
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, path)


def plot_sampling_counts(path: str | Path, rows: list[dict]) -> None:
    strata = sorted({row["stratum"] for row in rows})
    targets = sorted({row["target_short"] for row in rows})
    fig, ax = plt.subplots(figsize=(max(7.0, len(strata) * 0.8), 4.3))
    x = np.arange(len(strata))
    width = 0.8 / max(1, len(targets))
    for offset, target in enumerate(targets):
        vals = [sum(row["count"] for row in rows if row["target_short"] == target and row["stratum"] == stratum) for stratum in strata]
        ax.bar(x + (offset - len(targets) / 2) * width + width / 2, vals, width=width, label=target)
    ax.set_xticks(x)
    ax.set_xticklabels(strata, rotation=25, ha="right")
    ax.set_ylabel("selected patch count")
    ax.set_title("Stratified training/evaluation patch coverage")
    ax.legend()
    fig.tight_layout()
    _save(fig, path)


def milestone5_markdown_summary(metrics: list[dict], best_rows: list[dict], figure_paths: list[str]) -> str:
    lines = [
        "## Milestone 5: SOTA Numerical Training On K3 And Quintic",
        "",
        "Milestone 5 adds autodiff-trained invariant Kähler-potential corrections for both the Fermat quartic K3 and Fermat quintic threefold. The training path precomputes local Hessian tensors with JAX autodiff and optimizes geometric losses over invariant correction coefficients.",
        "",
        "Reported metrics use the Milestone 4 definitions. They are unweighted sample averages over valid positive local patches unless explicitly stated otherwise; sigma-style values remain sample-normalized volume-ratio proxies.",
        "",
        "### Selected Models",
        "",
        "| target | candidate | basis | loss | validation centered log-MA RMSE | validation sigma L1 | test centered log-MA RMSE | test sigma L1 |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in best_rows:
        lines.append(
            "| {target} | {candidate} | {basis} | {loss} | {val_rmse} | {val_sigma} | {test_rmse} | {test_sigma} |".format(
                target=row["target"],
                candidate=row["candidate"],
                basis=row["basis_set"],
                loss=row["loss_name"],
                val_rmse=_fmt(row.get("validation_centered_log_ma_rmse")),
                val_sigma=_fmt(row.get("validation_sigma_l1_volume_ratio")),
                test_rmse=_fmt(row.get("test_centered_log_ma_rmse")),
                test_sigma=_fmt(row.get("test_sigma_l1_volume_ratio")),
            )
        )
    lines.extend(
        [
            "",
            "### Benchmark Comparison",
            "",
            "| target | model | centered log-MA RMSE | sigma L1 volume ratio | positivity violation | valid patches |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in metrics:
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
    lines.extend(["", "### Figures", ""])
    for path in figure_paths:
        lines.append(f"- [{Path(path).name}]({path})")
    lines.extend(
        [
            "",
            "### Limitations",
            "",
            "This milestone is the first autodiff-training pass, not a final SOTA claim. The implemented model family is an invariant linear correction in a widened feature basis, not yet a full neural or graph architecture. It is designed to produce reliable numerical targets, ablations, and failure-mode evidence for a Milestone 6 probing loop.",
            "",
        ]
    )
    return "\n".join(lines)


def _save(fig: plt.Figure, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _safe_float(value: object) -> float:
    if value is None:
        return float("nan")
    return float(value)


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
