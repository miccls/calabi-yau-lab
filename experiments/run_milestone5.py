from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cy_expansion_lab.cy.fermat import FermatHypersurface
from cy_expansion_lab.cy.sampling import SampleSet, sample_fermat
from cy_expansion_lab.fit.autodiff_linear import (
    LossWeights,
    build_autodiff_linear_training_set,
    evaluate_autodiff_coefficients,
    train_autodiff_linear_potential,
)
from cy_expansion_lab.models.invariant_potential import InvariantPotentialModel, invariant_basis
from cy_expansion_lab.reporting.milestone4 import write_csv, write_json
from cy_expansion_lab.reporting.milestone5 import (
    milestone5_markdown_summary,
    plot_ablation_bars,
    plot_best_training_curves,
    plot_loss_components,
    plot_metric_bars,
    plot_model_size_vs_accuracy,
    plot_records_histogram,
    plot_residual_vs_p2,
    plot_sampling_counts,
    plot_stratum_residuals,
)
from cy_expansion_lab.validate.autodiff_geometry import fs_autodiff_benchmark, invariant_model_autodiff_benchmark
from cy_expansion_lab.validate.benchmark_metrics import metric_definition_rows


LITERATURE_REFRESH = [
    {
        "source": "CYJAX arXiv:2211.12520",
        "url": "https://arxiv.org/abs/2211.12520",
        "usable_idea": "JAX autodiff and algebraic Kahler-potential ansatz preserve Kahlerity and patch compatibility.",
        "milestone5_use": "Use JAX autodiff Hessians and keep a potential-based model rather than fitting arbitrary Hermitian tensors.",
        "comparison_status": "architecture inspiration; numeric metrics not copied because conventions must be matched table-by-table",
    },
    {
        "source": "CYJAX documentation",
        "url": "https://cyjax.readthedocs.io/",
        "usable_idea": "Single-projective-space Fermat/Dwork tools expose complex Hessians, induced metrics, sampling, Donaldson bases, and ML helpers.",
        "milestone5_use": "Keep local patch conventions explicit and use modular geometry/evaluation code.",
        "comparison_status": "implementation reference only",
    },
    {
        "source": "Symbolic Approximations via Extrinsic Symmetries arXiv:2412.19778",
        "url": "https://arxiv.org/abs/2412.19778",
        "usable_idea": "Extrinsic symmetries and special-locus analytic behavior can reduce model complexity and supply constraints.",
        "milestone5_use": "Include invariant features, phase invariance by construction, sqrt(s2)-style baseline, and document unsupported locus constraints.",
        "comparison_status": "theory-derived features/diagnostics; no numeric score copied",
    },
    {
        "source": "Interpretable Analytic Calabi-Yau Metrics via Symbolic Distillation arXiv:2602.07834",
        "url": "https://arxiv.org/abs/2602.07834",
        "usable_idea": "Low-order symmetric features such as p2 and sigma3 capture much of a determinant-ratio teacher on the Dwork quintic.",
        "milestone5_use": "Prioritize low-order symmetric bases and include model-size versus accuracy ablations.",
        "comparison_status": "symbolic-distillation context; teacher metric is not this repo's Ricci-flat benchmark",
    },
    {
        "source": "Sharp Edges arXiv:2606.26892",
        "url": "https://arxiv.org/abs/2606.26892",
        "usable_idea": "Symmetry handling, symmetry breaking in sampling, volume-ratio definitions, and failure modes matter for CY metric ML.",
        "milestone5_use": "Use stratified sampling, symmetry-aware features, volume-ratio metrics, and failure-mode plots.",
        "comparison_status": "methodology reference; no numeric score copied",
    },
    {
        "source": "Local FULLTEXT01.pdf",
        "url": "file:/Users/ms/Downloads/FULLTEXT01.pdf",
        "usable_idea": "The file exists locally, but available shell tools did not expose extractable text content in this environment.",
        "milestone5_use": "Treat prior user thesis context as important but do not block implementation on PDF extraction.",
        "comparison_status": "local text extraction unavailable; external Diva search snippet indicates symmetry constraints and near-SOTA models",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/experiments/milestone5_sota")
    parser.add_argument("--sample-count", type=int, default=96)
    parser.add_argument("--train-points", type=int, default=24)
    parser.add_argument("--val-points", type=int, default=18)
    parser.add_argument("--test-points", type=int, default=24)
    parser.add_argument("--maxiter", type=int, default=45)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--basis-sets", default="compact,rich,sqrt_s2_baseline")
    parser.add_argument("--losses", default="log_ma,hybrid_sigma")
    parser.add_argument("--seeds", default="")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    figures = Path("reports/figures")
    tables = Path("reports/tables")
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    targets = [
        ("k3", FermatHypersurface.quartic_k3(), args.seed),
        ("quintic", FermatHypersurface.quintic_threefold(), args.seed + 100),
    ]
    basis_sets = _filter_dict(_basis_sets(), args.basis_sets)
    loss_configs = _filter_dict(_loss_configs(), args.losses)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()] if args.seeds.strip() else [args.seed, args.seed + 1]

    all_sweep_rows: list[dict] = []
    best_rows: list[dict] = []
    benchmark_rows: list[dict] = []
    benchmark_records: dict[str, list[dict]] = {}
    best_histories: dict[str, list[dict]] = {}
    selected_records: dict[str, list[dict]] = {}
    sampling_rows: list[dict] = []
    p2_maps: dict[str, dict[int, float]] = {}

    for target_short, cy, target_seed in targets:
        samples = sample_fermat(cy, n_samples=args.sample_count, seed=target_seed, include_orbits=True)
        np.savez_compressed(out / f"{target_short}_samples.npz", z=samples.z, strata=samples.strata, split=samples.split)
        split_data = _split_samples(samples)
        p2_maps[target_short] = {
            int(i): float(value)
            for i, value in enumerate(invariant_basis(samples.z, ("p2",))[:, 0])
        }
        for split_name in ["train", "val", "test"]:
            labels, counts = np.unique([str(s).split(":")[0] for s in split_data[split_name]["strata"]], return_counts=True)
            sampling_rows.extend(
                {
                    "target_short": target_short,
                    "target": cy.name,
                    "split": split_name,
                    "stratum": str(label),
                    "count": int(count),
                }
                for label, count in zip(labels, counts)
            )

        initial_model = None
        if target_short == "k3":
            milestone3_path = Path("artifacts/models/fermat_quartic_milestone3_best.json")
            if milestone3_path.exists():
                initial_model = InvariantPotentialModel.load(milestone3_path)

        set_cache = {}
        candidates = []
        for basis_name, basis_names in basis_sets.items():
            train_set, val_set, test_set = _training_sets_for_basis(
                set_cache,
                target_short,
                basis_name,
                basis_names,
                cy,
                split_data,
                args,
            )
            for loss_name, weights in loss_configs.items():
                for seed in seeds:
                    model, metadata = train_autodiff_linear_potential(
                        train_set=train_set,
                        val_set=val_set,
                        cy=cy,
                        loss_weights=weights,
                        seed=int(seed),
                        maxiter=args.maxiter,
                        initial_model=initial_model,
                    )
                    train_eval = evaluate_autodiff_coefficients(train_set, model.coefficients, loss_weights=weights)
                    val_eval = evaluate_autodiff_coefficients(val_set, model.coefficients, loss_weights=weights)
                    test_eval = evaluate_autodiff_coefficients(test_set, model.coefficients, loss_weights=weights)
                    row = _candidate_row(
                        target_short=target_short,
                        cy=cy,
                        candidate=f"{target_short}_{basis_name}_{loss_name}_seed{seed}",
                        basis_name=basis_name,
                        loss_name=loss_name,
                        seed=int(seed),
                        basis_size=len(basis_names),
                        metadata=metadata,
                        train_eval=train_eval,
                        val_eval=val_eval,
                        test_eval=test_eval,
                    )
                    all_sweep_rows.append(row)
                    candidates.append((row, model, metadata, test_eval))

        selected = min(
            candidates,
            key=lambda item: (
                item[0]["validation_positivity_violation_rate"],
                item[0]["validation_centered_log_ma_rmse"],
                item[0]["validation_sigma_l1_volume_ratio"],
            ),
        )
        best_row, best_model, best_metadata, best_test_eval = selected
        best_row["selected"] = True
        for row in all_sweep_rows:
            if row["candidate"] == best_row["candidate"]:
                row["selected"] = True
            else:
                row.setdefault("selected", False)
        best_path = Path(f"artifacts/models/fermat_{'quartic' if target_short == 'k3' else 'quintic'}_milestone5_best.json")
        best_metadata["milestone5_selection"] = best_row
        best_metadata["model_path"] = str(best_path)
        best_metadata["sample_count"] = int(args.sample_count)
        best_metadata["literature_refresh"] = LITERATURE_REFRESH
        best_model.metadata = _compact_metadata(best_metadata)
        best_model.save(best_path)
        best_row["checkpoint"] = str(best_path)
        best_rows.append(best_row)
        best_histories[target_short] = best_metadata.get("history", [])
        selected_records[f"{target_short}_milestone5_best"] = best_test_eval["records"]

        _add_benchmark(
            benchmark_rows,
            benchmark_records,
            "fs",
            fs_autodiff_benchmark(
                split_data["test"]["z"],
                split_data["test"]["strata"],
                cy=cy,
                max_points=args.test_points,
                ricci_points=1,
            ),
            target_short,
        )
        if target_short == "k3":
            for label, path in [
                ("milestone2", Path("artifacts/models/fermat_quartic_invariant_potential.json")),
                ("milestone3", Path("artifacts/models/fermat_quartic_milestone3_best.json")),
            ]:
                if path.exists():
                    baseline = InvariantPotentialModel.load(path)
                    _add_benchmark(
                        benchmark_rows,
                        benchmark_records,
                        label,
                        invariant_model_autodiff_benchmark(
                            split_data["test"]["z"],
                            split_data["test"]["strata"],
                            cy=cy,
                            model=baseline,
                            model_name=f"{target_short}_{label}",
                            max_points=args.test_points,
                            ricci_points=1,
                        ),
                        target_short,
                    )
        _add_precomputed_benchmark(
            benchmark_rows,
            benchmark_records,
            label="milestone5_best",
            target_short=target_short,
            target=cy.name,
            model_name=f"{target_short}_milestone5_best",
            payload=best_test_eval,
        )

    all_records = [record | {"record_group": label} for label, records in benchmark_records.items() for record in records]
    all_selected_records = [
        record | {"record_group": label} for label, records in selected_records.items() for record in records
    ]
    summary = {
        "run": {
            "command": (
                "uv run --extra dev python experiments/run_milestone5.py "
                f"--sample-count {args.sample_count} --train-points {args.train_points} "
                f"--val-points {args.val_points} --test-points {args.test_points} --maxiter {args.maxiter} "
                f"--basis-sets {args.basis_sets} --losses {args.losses}"
            ),
            "output_dir": str(out),
            "seed": int(args.seed),
            "sample_count": int(args.sample_count),
            "train_points": int(args.train_points),
            "val_points": int(args.val_points),
            "test_points": int(args.test_points),
            "maxiter": int(args.maxiter),
            "basis_sets": list(basis_sets),
            "losses": list(loss_configs),
            "seeds": seeds,
        },
        "best": best_rows,
        "benchmark_metrics": benchmark_rows,
        "sweep_rows": all_sweep_rows,
        "metric_definitions": metric_definition_rows(),
        "literature_refresh": LITERATURE_REFRESH,
    }

    write_json(out / "milestone5_summary.json", summary)
    write_csv(out / "milestone5_sweep.csv", all_sweep_rows)
    write_csv(out / "milestone5_benchmark_metrics.csv", benchmark_rows)
    write_json(out / "milestone5_pointwise_records.json", all_records)
    write_json(out / "milestone5_selected_pointwise_records.json", all_selected_records)
    write_csv(out / "milestone5_literature_refresh.csv", LITERATURE_REFRESH)
    write_csv(out / "milestone5_sampling_counts.csv", sampling_rows)

    write_csv(tables / "milestone5_sweep.csv", all_sweep_rows)
    write_csv(tables / "milestone5_benchmark_metrics.csv", benchmark_rows)
    write_csv(tables / "milestone5_literature_refresh.csv", LITERATURE_REFRESH)
    write_csv(tables / "milestone5_sampling_counts.csv", sampling_rows)

    figure_paths = _make_figures(
        figures=figures,
        sweep_rows=all_sweep_rows,
        benchmark_rows=benchmark_rows,
        benchmark_records=benchmark_records,
        selected_records=selected_records,
        best_histories=best_histories,
        sampling_rows=sampling_rows,
        p2_maps=p2_maps,
    )
    section = milestone5_markdown_summary(benchmark_rows, best_rows, figure_paths)
    Path("reports/milestone5_section.md").write_text(section, encoding="utf-8")
    Path("reports/milestone5_log.md").write_text("# Milestone 5 Run Log\n\n" + section, encoding="utf-8")
    _append_section("reports/paper.md", "## Milestone 5: SOTA Numerical Training On K3 And Quintic", section)
    _append_research_log(best_rows, benchmark_rows)
    _update_tex(best_rows)
    print(section)


def _basis_sets() -> dict[str, tuple[str, ...]]:
    return {
        "compact": ("p2", "p3", "e2", "e3", "centered_l2", "sqrt_centered_l2"),
        "rich": (
            "p2",
            "p3",
            "p4",
            "p5",
            "p6",
            "e2",
            "e3",
            "e4",
            "centered_l2",
            "centered_l3",
            "centered_l4",
            "entropy",
        ),
        "extended_theory": (
            "p2",
            "p3",
            "p4",
            "p5",
            "p6",
            "p7",
            "p8",
            "e2",
            "e3",
            "e4",
            "e5",
            "centered_l2",
            "centered_l3",
            "centered_l4",
            "centered_l5",
            "centered_l6",
            "sqrt_centered_l2",
            "entropy",
        ),
        "sqrt_s2_baseline": ("sqrt_centered_l2", "centered_l2", "p2", "e2"),
    }


def _loss_configs() -> dict[str, LossWeights]:
    return {
        "log_ma": LossWeights(centered_log_ma=1.0, volume_ratio_mse=0.0, sigma_l1_smooth=0.0, positivity=25.0, l2=1e-4),
        "hybrid_sigma": LossWeights(centered_log_ma=0.7, volume_ratio_mse=0.2, sigma_l1_smooth=0.1, positivity=35.0, l2=3e-4),
    }


def _filter_dict(payload: dict, names: str) -> dict:
    requested = [name.strip() for name in names.split(",") if name.strip()]
    missing = [name for name in requested if name not in payload]
    if missing:
        raise ValueError(f"Unknown option(s): {missing}; available: {sorted(payload)}")
    return {name: payload[name] for name in requested} if requested else payload


def _split_samples(samples: SampleSet) -> dict[str, dict[str, np.ndarray]]:
    return {
        split: {
            "z": samples.z[samples.split == split],
            "strata": samples.strata[samples.split == split],
        }
        for split in ["train", "val", "test"]
    }


def _training_sets_for_basis(cache: dict, target_short: str, basis_name: str, basis_names: tuple[str, ...], cy, split_data, args):
    key = (target_short, basis_name)
    if key not in cache:
        cache[key] = (
            build_autodiff_linear_training_set(
                split_data["train"]["z"],
                split_data["train"]["strata"],
                cy,
                basis_names,
                max_points=args.train_points,
                selection="stratified",
            ),
            build_autodiff_linear_training_set(
                split_data["val"]["z"],
                split_data["val"]["strata"],
                cy,
                basis_names,
                max_points=args.val_points,
                selection="stratified",
            ),
            build_autodiff_linear_training_set(
                split_data["test"]["z"],
                split_data["test"]["strata"],
                cy,
                basis_names,
                max_points=args.test_points,
                selection="stratified",
            ),
        )
    return cache[key]


def _candidate_row(
    target_short: str,
    cy: FermatHypersurface,
    candidate: str,
    basis_name: str,
    loss_name: str,
    seed: int,
    basis_size: int,
    metadata: dict,
    train_eval: dict,
    val_eval: dict,
    test_eval: dict,
) -> dict:
    train_summary = train_eval["summary"]
    val_summary = val_eval["summary"]
    test_summary = test_eval["summary"]
    train_loss = train_eval["loss"]
    return {
        "target_short": target_short,
        "target": cy.name,
        "candidate": candidate,
        "basis_set": basis_name,
        "loss_name": loss_name,
        "seed": seed,
        "basis_size": basis_size,
        "selected": False,
        "train_centered_log_ma_rmse": train_summary["centered_log_ma_rmse"],
        "validation_centered_log_ma_rmse": val_summary["centered_log_ma_rmse"],
        "test_centered_log_ma_rmse": test_summary["centered_log_ma_rmse"],
        "train_sigma_l1_volume_ratio": train_summary["sigma_l1_volume_ratio"],
        "validation_sigma_l1_volume_ratio": val_summary["sigma_l1_volume_ratio"],
        "test_sigma_l1_volume_ratio": test_summary["sigma_l1_volume_ratio"],
        "train_positivity_violation_rate": train_summary["positivity_violation_rate"],
        "validation_positivity_violation_rate": val_summary["positivity_violation_rate"],
        "test_positivity_violation_rate": test_summary["positivity_violation_rate"],
        "test_min_eigenvalue_min": test_summary["min_eigenvalue_min"],
        "optimizer_success": metadata["optimizer_success"],
        "optimizer_iterations": metadata["optimizer_iterations"],
        "train_centered_log_ma_mse": train_loss["centered_log_ma_mse"],
        "train_volume_ratio_mse": train_loss["volume_ratio_mse"],
        "train_sigma_l1_smooth": train_loss["sigma_l1_smooth"],
        "train_positivity_penalty": train_loss["positivity_penalty"],
    }


def _add_benchmark(rows: list[dict], records: dict[str, list[dict]], label: str, payload: dict, target_short: str) -> None:
    summary = dict(payload["summary"])
    summary["target_short"] = target_short
    summary["benchmark_label"] = label
    rows.append(summary)
    records[f"{target_short}_{label}"] = payload["records"]


def _add_precomputed_benchmark(
    rows: list[dict],
    records: dict[str, list[dict]],
    label: str,
    target_short: str,
    target: str,
    model_name: str,
    payload: dict,
) -> None:
    summary = dict(payload["summary"])
    summary.update(
        {
            "target": target,
            "target_short": target_short,
            "model": model_name,
            "benchmark_label": label,
            "sampled_count": summary.get("evaluated_count"),
            "skipped_count": 0,
            "autodiff_backend": "jax_precomputed_basis_hessians",
            "metric_normalization": "unweighted sample averages over valid positive local patches; volume ratio scale fitted per evaluated model/sample set",
        }
    )
    rows.append(summary)
    records[f"{target_short}_{label}"] = payload["records"]


def _make_figures(
    figures: Path,
    sweep_rows: list[dict],
    benchmark_rows: list[dict],
    benchmark_records: dict[str, list[dict]],
    selected_records: dict[str, list[dict]],
    best_histories: dict[str, list[dict]],
    sampling_rows: list[dict],
    p2_maps: dict[str, dict[int, float]],
) -> list[str]:
    paths = [
        figures / "milestone5_training_curves.png",
        figures / "milestone5_loss_components.png",
        figures / "milestone5_centered_log_ma_bars.png",
        figures / "milestone5_sigma_l1_bars.png",
        figures / "milestone5_residual_histograms.png",
        figures / "milestone5_volume_ratio_histograms.png",
        figures / "milestone5_min_eigen_histograms.png",
        figures / "milestone5_stratum_residuals.png",
        figures / "milestone5_model_size_vs_accuracy.png",
        figures / "milestone5_basis_ablation.png",
        figures / "milestone5_loss_ablation.png",
        figures / "milestone5_sampling_counts.png",
        figures / "milestone5_k3_residual_vs_p2.png",
        figures / "milestone5_quintic_residual_vs_p2.png",
    ]
    plot_best_training_curves(paths[0], best_histories)
    plot_loss_components(paths[1], sweep_rows)
    plot_metric_bars(paths[2], benchmark_rows, "centered_log_ma_rmse", "centered log-MA RMSE, dimensionless", "Milestone 5 benchmark comparison")
    plot_metric_bars(paths[3], benchmark_rows, "sigma_l1_volume_ratio", "sigma-style L1 volume-ratio error", "Milestone 5 sigma-style comparison")
    plot_records_histogram(paths[4], benchmark_records, "centered_log_ma_residual", "|centered log-MA residual|", "Benchmark residual histograms")
    plot_records_histogram(paths[5], benchmark_records, "volume_ratio", "raw volume ratio", "Raw volume-ratio histograms")
    plot_records_histogram(paths[6], benchmark_records, "min_eigenvalue", "minimum metric eigenvalue", "Metric positivity distributions")
    plot_stratum_residuals(paths[7], benchmark_records)
    plot_model_size_vs_accuracy(paths[8], sweep_rows)
    plot_ablation_bars(paths[9], sweep_rows, "basis_set", "validation_centered_log_ma_rmse", "Basis ablation: best validation RMSE")
    plot_ablation_bars(paths[10], sweep_rows, "loss_name", "validation_centered_log_ma_rmse", "Loss ablation: best validation RMSE")
    plot_sampling_counts(paths[11], sampling_rows)
    plot_residual_vs_p2(paths[12], selected_records.get("k3_milestone5_best", []), p2_maps.get("k3", {}), "K3 best residuals versus p2")
    plot_residual_vs_p2(paths[13], selected_records.get("quintic_milestone5_best", []), p2_maps.get("quintic", {}), "Quintic best residuals versus p2")
    return [str(path.relative_to("reports")) for path in paths]


def _compact_metadata(metadata: dict) -> dict:
    compact = dict(metadata)
    if "history" in compact:
        compact["history_length"] = len(compact.pop("history"))
    return compact


def _append_section(path: str, marker: str, section: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8") if target.exists() else ""
    if marker in text:
        text = text[: text.index(marker)].rstrip()
    target.write_text(text.rstrip() + "\n\n" + section + "\n", encoding="utf-8")


def _append_research_log(best_rows: list[dict], benchmark_rows: list[dict]) -> None:
    lines = [
        "## Milestone 5 Run",
        "",
        "- Command: `uv run --extra dev python experiments/run_milestone5.py --sample-count 96 --train-points 18 --val-points 14 --test-points 18 --maxiter 30 --basis-sets compact,rich,sqrt_s2_baseline --losses log_ma,hybrid_sigma --seeds 20260707,20260708`.",
        "- Ran autodiff-linear invariant correction sweeps for Fermat quartic K3 and Fermat quintic.",
        "- Selected best models by validation positivity violation rate, centered log-Monge-Ampere RMSE, and sigma-style volume-ratio error.",
    ]
    for row in best_rows:
        lines.append(
            f"- Best {row['target']}: `{row['candidate']}` at `{row['checkpoint']}`; "
            f"test centered log-MA RMSE `{row['test_centered_log_ma_rmse']}`, "
            f"test sigma-style L1 `{row['test_sigma_l1_volume_ratio']}`."
        )
    lines.append("- Generated Milestone 5 figures under `reports/figures/milestone5_*.png` and tables under `reports/tables/milestone5_*.csv`.")
    lines.append("- Published comparisons remain convention-aware; no external numeric table values were copied into repo comparisons.")
    entry = "\n".join(lines) + "\n"
    Path("reports/milestone5_log.md").write_text(entry, encoding="utf-8")
    log_path = Path("reports/research_log.md")
    text = log_path.read_text(encoding="utf-8") if log_path.exists() else "# Research Log\n"
    if "## Milestone 5 Run" in text:
        text = text[: text.index("## Milestone 5 Run")].rstrip()
    log_path.write_text(text.rstrip() + "\n\n" + entry, encoding="utf-8")


def _update_tex(best_rows: list[dict]) -> None:
    path = Path("reports/paper.tex")
    if not path.exists():
        return
    rows = "\n".join(
        f"{row['target']} & {row['candidate']} & {row['test_centered_log_ma_rmse']:.6g} & {row['test_sigma_l1_volume_ratio']:.6g} \\\\"
        for row in best_rows
    )
    section = rf"""
\section{{Milestone 5 Autodiff Training}}
Milestone 5 introduces autodiff-trained invariant Kähler-potential corrections for both the Fermat quartic K3 and Fermat quintic. Local Hessian tensors for invariant basis terms are computed with JAX and the coefficient optimization uses geometric losses derived from the Milestone 4 metric suite.

\begin{{center}}
\begin{{tabular}}{{llll}}
target & candidate & centered log-MA RMSE & sigma-style $L^1$ \\
{rows}
\end{{tabular}}
\end{{center}}

The reported sigma-style values are sample-normalized volume-ratio proxies. They are not direct claims of matching any published table unless the sampling measure and normalization are later aligned.
"""
    text = path.read_text(encoding="utf-8")
    marker = "\\section{Milestone 5 Autodiff Training}"
    if marker in text:
        text = text[: text.index(marker)].rstrip() + "\n\\end{document}\n"
    text = text.replace("\\end{document}", section + "\n\\end{document}")
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
