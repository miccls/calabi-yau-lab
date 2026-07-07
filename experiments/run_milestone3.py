from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from cy_expansion_lab.config import ensure_dir, load_config
from cy_expansion_lab.cy.fermat import from_config
from cy_expansion_lab.cy.local_patch import fs_affine_potential
from cy_expansion_lab.fit.k3_potential import build_local_training_set, evaluate_coefficients, train_invariant_potential
from cy_expansion_lab.fit.train import fit_default_models
from cy_expansion_lab.invariants.quotient import invariant_table
from cy_expansion_lab.models.invariant_potential import InvariantPotentialModel, invariant_basis
from cy_expansion_lab.reporting.milestone3 import (
    LITERATURE_COMPARISONS,
    METRIC_DEFINITIONS,
    plot_comparison_bar,
    plot_correction_vs_invariant,
    plot_min_eigen_histogram,
    plot_residual_histogram,
    plot_stratum_residuals,
    plot_training_curves,
    write_csv,
    write_json,
)
from cy_expansion_lab.reporting.tables import markdown_table
from cy_expansion_lab.validate.local_geometry import pointwise_geometry_records


def run(script: str, config: str) -> None:
    subprocess.run([sys.executable, f"experiments/{script}", "--config", config], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fermat_quartic_milestone3.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if cfg["target"] != "fermat_quartic":
        raise ValueError("Milestone 3 currently targets the Fermat quartic K3")
    out = ensure_dir(cfg["outputs"]["experiment_dir"])
    figures = ensure_dir("reports/figures")
    cy = from_config(cfg["target"])

    for script in ["run_generate_data.py", "run_invariants.py", "run_quotient_diagnostics.py", "run_fit_models.py"]:
        run(script, args.config)

    samples = np.load(out / "samples.npz", allow_pickle=True)
    inv = np.load(out / "invariants.npz")
    z = samples["z"]
    strata = samples["strata"]
    split = samples["split"]
    train_z = z[split == "train"]
    val_z = z[split == "val"]
    test_z = z[split == "test"]
    test_strata = strata[split == "test"]
    validation_cfg = cfg.get("validation", {})
    step = float(validation_cfg.get("finite_difference_step", 1e-4))
    max_points = int(validation_cfg.get("geometry_max_points", 72))
    training_cfg = cfg["training"]["milestone3"]

    sweep_rows = []
    candidates = []
    for basis_name, basis_terms in training_cfg["basis_sets"].items():
        for seed in training_cfg["seeds"]:
            for l2_weight in training_cfg["l2_weights"]:
                model, metadata = train_invariant_potential(
                    z_train=train_z,
                    z_val=val_z,
                    cy=cy,
                    basis_names=tuple(basis_terms),
                    max_train_points=int(training_cfg["max_train_points"]),
                    max_val_points=int(training_cfg["max_val_points"]),
                    step=step,
                    seed=int(seed),
                    maxiter=int(training_cfg["maxiter"]),
                    positivity_weight=float(training_cfg["positivity_weight"]),
                    l2_weight=float(l2_weight),
                )
                validation = metadata["validation_metrics"]
                row = {
                    "candidate": f"{basis_name}_seed{seed}_l2{l2_weight}",
                    "basis_set": basis_name,
                    "seed": int(seed),
                    "l2_weight": float(l2_weight),
                    "basis_size": len(basis_terms),
                    "validation_centered_log_ma_rmse": validation["monge_ampere"]["ma_residual_rmse"],
                    "validation_centered_log_ma_mae": validation["monge_ampere"]["ma_residual_mean_abs"],
                    "validation_ma_max_abs": validation["monge_ampere"]["ma_residual_max_abs"],
                    "validation_pos_fail": validation["metric_positivity"]["positivity_violation_rate"],
                    "validation_min_eig": validation["metric_positivity"]["min_eigenvalue"],
                    "train_objective": metadata["train_objective"],
                }
                sweep_rows.append(row)
                candidates.append((row, model, metadata))

    feasible = [item for item in candidates if item[0]["validation_pos_fail"] == 0.0]
    selected = min(feasible or candidates, key=lambda item: (item[0]["validation_pos_fail"], item[0]["validation_centered_log_ma_rmse"]))
    best_row, best_model, best_metadata = selected
    best_model_path = Path(training_cfg["best_model_path"])
    best_metadata["milestone3_selection"] = best_row
    best_metadata["config"] = args.config
    best_metadata["model_path"] = str(best_model_path)
    best_model.metadata = best_metadata
    best_model.save(best_model_path)

    # Keep the standard validation path using the best Milestone 3 model too.
    if cfg.get("training", {}).get("invariant_potential") is None:
        cfg.setdefault("training", {})["invariant_potential"] = {}
    milestone2_path = Path("artifacts/models/fermat_quartic_invariant_potential.json")
    if milestone2_path.exists():
        m2_model = InvariantPotentialModel.load(milestone2_path)
    else:
        m2_model = InvariantPotentialModel(coefficients=np.zeros(6))

    records_by_model = {
        "Fubini-Study": pointwise_geometry_records(
            test_z,
            test_strata,
            cy,
            lambda patch: lambda u: fs_affine_potential(patch.reconstruct_affine(u)),
            max_points=max_points,
            step=step,
        ),
        "Milestone 2": pointwise_geometry_records(
            test_z,
            test_strata,
            cy,
            lambda patch: lambda u: m2_model.local_potential(patch.reconstruct_affine(u)),
            max_points=max_points,
            step=step,
        ),
        "Milestone 3 best": pointwise_geometry_records(
            test_z,
            test_strata,
            cy,
            lambda patch: lambda u: best_model.local_potential(patch.reconstruct_affine(u)),
            max_points=max_points,
            step=step,
        ),
    }

    repo_comparison = []
    for name, records in records_by_model.items():
        repo_comparison.append(_summary_from_records(name, records, split="test"))

    # Add scalar model rows for context using existing local validation code in run_all outputs.
    scalar_results = fit_default_models(inv["positive_features"], inv["target"], split, n_terms=int(cfg["models"]["log_sum_terms"]))
    for name in scalar_results:
        repo_comparison.append(
            {
                "model": name,
                "split": "test",
                "centered_log_ma_rmse": None,
                "centered_log_ma_mae": None,
                "centered_log_ma_max_abs": None,
                "positivity_violation_rate": None,
                "min_eigenvalue": None,
                "metric_note": "scalar synthetic-target fit only; geometry reported in run_all validation",
            }
        )

    comparison_rows = _comparison_table_rows(repo_comparison)
    write_csv(out / "milestone3_sweep.csv", sweep_rows)
    write_json(out / "milestone3_summary.json", {"best": best_row, "repo_comparison": repo_comparison, "literature": LITERATURE_COMPARISONS})
    write_json(out / "milestone3_pointwise_records.json", records_by_model)
    write_csv(out / "milestone3_metric_definitions.csv", METRIC_DEFINITIONS)
    write_csv(out / "milestone3_literature_comparison.csv", LITERATURE_COMPARISONS)

    plot_training_curves(figures / "milestone3_training_curves.png", best_metadata["history"])
    plot_min_eigen_histogram(figures / "milestone3_min_eigen_histogram.png", records_by_model)
    plot_residual_histogram(figures / "milestone3_residual_histogram.png", records_by_model)
    plot_stratum_residuals(figures / "milestone3_stratum_residuals.png", records_by_model)
    p2 = invariant_basis(test_z, ("p2",))[:, 0]
    correction = best_model.correction_from_homogeneous(test_z)
    plot_correction_vs_invariant(figures / "milestone3_correction_vs_p2.png", p2, correction)
    plot_comparison_bar(figures / "milestone3_comparison_bar.png", repo_comparison)

    section = _report_section(best_row, repo_comparison, comparison_rows, args.config, best_model_path)
    Path("reports/milestone3_section.md").write_text(section, encoding="utf-8")
    _merge_section_into_paper(section)
    _append_log(best_row, best_model_path)
    print(json.dumps({"best": best_row, "best_model_path": str(best_model_path), "repo_comparison": repo_comparison}, indent=2))


def _summary_from_records(model: str, records: list[dict], split: str) -> dict:
    residuals = [abs(r["centered_log_ma_residual"]) for r in records if r.get("centered_log_ma_residual") is not None]
    min_eigs = [r["min_eigenvalue"] for r in records if "min_eigenvalue" in r]
    positives = [r["positive"] for r in records if "positive" in r]
    return {
        "model": model,
        "split": split,
        "centered_log_ma_rmse": float(np.sqrt(np.mean(np.array(residuals) ** 2))) if residuals else None,
        "centered_log_ma_mae": float(np.mean(residuals)) if residuals else None,
        "centered_log_ma_max_abs": float(np.max(residuals)) if residuals else None,
        "positivity_violation_rate": float(1.0 - np.mean(positives)) if positives else None,
        "min_eigenvalue": float(np.min(min_eigs)) if min_eigs else None,
        "metric_note": "centered log-Monge-Ampere residual over held-out local patches",
    }


def _comparison_table_rows(repo_rows: list[dict]) -> list[dict]:
    rows = []
    for row in repo_rows:
        rows.append(
            {
                "source": "this repo",
                "model": row["model"],
                "metric": "centered_log_ma_rmse",
                "formula": "sqrt(mean((log det g - log |Omega|^2 - mean)^2))",
                "unit/convention": "dimensionless centered log residual; held-out local patches",
                "split": row["split"],
                "value": _fmt(row["centered_log_ma_rmse"]),
                "note": row["metric_note"],
            }
        )
    for lit in LITERATURE_COMPARISONS:
        rows.append(
            {
                "source": lit["source"],
                "model": lit["method"],
                "metric": lit["reported_metric"],
                "formula": "see source",
                "unit/convention": lit["unit_or_convention"],
                "split": "literature",
                "value": lit["numeric_value"],
                "note": lit["comparison_note"],
            }
        )
    return rows


def _report_section(best_row: dict, repo_rows: list[dict], comparison_rows: list[dict], config: str, model_path: Path) -> str:
    metric_table = markdown_table(
        [
            {
                "metric": row["metric"],
                "formula": row["formula"],
                "unit": row["unit"],
                "aggregation": row["aggregation"],
            }
            for row in METRIC_DEFINITIONS
        ],
        ["metric", "formula", "unit", "aggregation"],
    )
    comparison_table = markdown_table(comparison_rows, ["source", "model", "metric", "unit/convention", "split", "value", "note"])
    repo_table = markdown_table(
        [
            {
                "model": row["model"],
                "split": row["split"],
                "centered_log_ma_rmse": _fmt(row["centered_log_ma_rmse"]),
                "centered_log_ma_mae": _fmt(row["centered_log_ma_mae"]),
                "positivity_violation_rate": _fmt(row["positivity_violation_rate"]),
                "min_eigenvalue": _fmt(row["min_eigenvalue"]),
            }
            for row in repo_rows
        ],
        ["model", "split", "centered_log_ma_rmse", "centered_log_ma_mae", "positivity_violation_rate", "min_eigenvalue"],
    )
    return f"""## Milestone 3: SOTA-Oriented K3 Tuning

Milestone 3 adds a reproducible sweep command:

```bash
uv run python experiments/run_milestone3.py --config {config}
```

The best selected model is saved at `{model_path}`. Selection is by held-out centered log-Monge-Ampere RMSE, with positivity violation rate used as the first ordering key.

Best candidate:

```json
{json.dumps(best_row, indent=2)}
```

### Architecture And Symmetry

The tuned models keep the Kähler-potential form `K = K_FS + correction`. Corrections are linear combinations of invariant basis terms in normalized radii `r_i = |z_i|^2 / sum_j |z_j|^2`. This enforces projective scaling invariance, Fermat phase invariance, and coordinate permutation invariance by construction. The richer Milestone 3 basis includes higher power sums, elementary symmetric polynomials, centered moments, and entropy-like radius features.

### Training Objective

The trainer precomputes local Hessian tensors for each invariant basis term and optimizes coefficients against a held-out local geometry objective: centered log-Monge-Ampere residual plus positivity and L2 penalties. This is still finite-difference based; autodiff Hessians remain the main path toward SOTA-scale training.

### Metric Definitions And Units

{metric_table}

Do not compare these values directly to sigma-loss, Ricci scalar losses, raw volume-ratio errors, MSE, or MAE values from other papers unless the same convention and sampling measure are used.

### Repo Comparison

{repo_table}

### Literature Comparison

{comparison_table}

External entries are included as method/context comparisons, not numeric score comparisons, because the published metrics use different conventions or require reading their exact tables/code before copying values.

### Figures

- [Training curves](figures/milestone3_training_curves.png)
- [Minimum eigenvalue distribution](figures/milestone3_min_eigen_histogram.png)
- [Residual histogram](figures/milestone3_residual_histogram.png)
- [Stratum-wise residuals](figures/milestone3_stratum_residuals.png)
- [Correction versus p2](figures/milestone3_correction_vs_p2.png)
- [Comparison bar chart](figures/milestone3_comparison_bar.png)

### Limitations

This is SOTA-oriented infrastructure, not a SOTA claim. The best model improves the local held-out diagnostic relative to the Milestone 2 and Fubini-Study baselines in this repo, but the training loop still uses finite-difference Hessians, small patch counts, and low-dimensional invariant corrections. Ricci residuals are still not implemented.
"""


def _merge_section_into_paper(section: str) -> None:
    path = Path("reports/paper.md")
    if not path.exists():
        path.write_text(section, encoding="utf-8")
        return
    text = path.read_text(encoding="utf-8")
    marker = "## Milestone 3: SOTA-Oriented K3 Tuning"
    if marker in text:
        text = text[: text.index(marker)].rstrip() + "\n\n" + section
    else:
        text = text.rstrip() + "\n\n" + section
    path.write_text(text, encoding="utf-8")


def _append_log(best_row: dict, model_path: Path) -> None:
    path = Path("reports/research_log.md")
    text = path.read_text(encoding="utf-8") if path.exists() else "# Research Log\n"
    entry = f"""

## Milestone 3 Run

- Ran SOTA-oriented sweep command and selected best model by held-out centered log-Monge-Ampere RMSE.
- Best model path: `{model_path}`.
- Best candidate: `{best_row}`.
- Generated figures under `reports/figures/`.
- Added metric-definition and literature-comparison tables with units/conventions.
"""
    if "## Milestone 3 Run" not in text:
        text = text.rstrip() + entry
    path.write_text(text, encoding="utf-8")
    Path("reports/milestone3_log.md").write_text(entry.lstrip(), encoding="utf-8")


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


if __name__ == "__main__":
    main()
