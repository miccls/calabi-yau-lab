from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cy_expansion_lab.cy.fermat import FermatHypersurface
from cy_expansion_lab.cy.sampling import sample_fermat
from cy_expansion_lab.models.invariant_potential import InvariantPotentialModel
from cy_expansion_lab.reporting.milestone4 import (
    markdown_summary,
    plot_k3_quintic_baseline,
    plot_metric_bars,
    plot_min_eigen_histograms,
    plot_residual_histograms,
    write_csv,
    write_json,
)
from cy_expansion_lab.research.theory_registry import theory_registry_rows, write_theory_registry
from cy_expansion_lab.validate.autodiff_geometry import (
    fs_autodiff_benchmark,
    invariant_model_autodiff_benchmark,
)
from cy_expansion_lab.validate.benchmark_metrics import metric_definition_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/experiments/milestone4_benchmark")
    parser.add_argument("--sample-count", type=int, default=120)
    parser.add_argument("--max-points", type=int, default=36)
    parser.add_argument("--ricci-points", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260707)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_tables = Path("reports/tables")
    report_figures = Path("reports/figures")
    report_tables.mkdir(parents=True, exist_ok=True)
    report_figures.mkdir(parents=True, exist_ok=True)

    k3 = FermatHypersurface.quartic_k3()
    quintic = FermatHypersurface.quintic_threefold()
    k3_samples = sample_fermat(k3, n_samples=args.sample_count, seed=args.seed, include_orbits=True)
    quintic_samples = sample_fermat(quintic, n_samples=args.sample_count, seed=args.seed + 1, include_orbits=True)

    np.savez_compressed(out / "k3_samples.npz", z=k3_samples.z, strata=k3_samples.strata, split=k3_samples.split)
    np.savez_compressed(
        out / "quintic_samples.npz",
        z=quintic_samples.z,
        strata=quintic_samples.strata,
        split=quintic_samples.split,
    )

    results = []
    records_by_model: dict[str, list[dict]] = {}

    def add_result(label: str, payload: dict) -> None:
        summary = payload["summary"]
        results.append(summary)
        records_by_model[label] = payload["records"]

    add_result(
        "k3_fs",
        fs_autodiff_benchmark(
            k3_samples.z,
            k3_samples.strata,
            cy=k3,
            max_points=args.max_points,
            ricci_points=args.ricci_points,
        ),
    )
    for model_name, path in [
        ("k3_milestone2_invariant", Path("artifacts/models/fermat_quartic_invariant_potential.json")),
        ("k3_milestone3_best", Path("artifacts/models/fermat_quartic_milestone3_best.json")),
    ]:
        if path.exists():
            model = InvariantPotentialModel.load(path)
            add_result(
                model_name,
                invariant_model_autodiff_benchmark(
                    k3_samples.z,
                    k3_samples.strata,
                    cy=k3,
                    model=model,
                    model_name=model_name,
                    max_points=args.max_points,
                    ricci_points=args.ricci_points,
                ),
            )
    add_result(
        "quintic_fs",
        fs_autodiff_benchmark(
            quintic_samples.z,
            quintic_samples.strata,
            cy=quintic,
            max_points=args.max_points,
            ricci_points=args.ricci_points,
        ),
    )

    all_records = [record for records in records_by_model.values() for record in records]
    metric_rows = metric_definition_rows()
    theory_rows = theory_registry_rows()
    summary = {
        "run": {
            "output_dir": str(out),
            "seed": args.seed,
            "sample_count": args.sample_count,
            "max_points": args.max_points,
            "ricci_points": args.ricci_points,
            "normalization": (
                "All aggregate metrics are unweighted sample averages over valid positive local patches. "
                "Volume-ratio and sigma-style metrics fit one scale kappa per evaluated model/sample set."
            ),
        },
        "sample_metadata": {
            "k3": k3_samples.metadata,
            "quintic": quintic_samples.metadata,
        },
        "metrics": results,
        "metric_definitions": metric_rows,
        "theory_registry": theory_rows,
    }

    write_json(out / "milestone4_summary.json", summary)
    write_csv(out / "milestone4_metrics.csv", results)
    write_json(out / "milestone4_pointwise_records.json", all_records)
    write_csv(out / "milestone4_pointwise_records.csv", all_records)
    write_csv(out / "milestone4_metric_definitions.csv", metric_rows)
    write_json(out / "milestone4_theory_registry.json", theory_rows)
    write_theory_registry(out / "theory_registry.json")

    write_csv(report_tables / "milestone4_metrics.csv", results)
    write_csv(report_tables / "milestone4_metric_definitions.csv", metric_rows)
    write_csv(report_tables / "milestone4_theory_registry.csv", theory_rows)

    plot_metric_bars(
        report_figures / "milestone4_centered_log_ma_rmse.png",
        results,
        metric="centered_log_ma_rmse",
        ylabel="centered log-Monge-Ampere RMSE, dimensionless",
        title="Milestone 4 autodiff benchmark",
    )
    plot_metric_bars(
        report_figures / "milestone4_sigma_l1_volume_ratio.png",
        results,
        metric="sigma_l1_volume_ratio",
        ylabel="sigma-style L1 volume-ratio error, dimensionless",
        title="Sample-normalized volume-ratio error",
    )
    plot_residual_histograms(report_figures / "milestone4_residual_histograms.png", records_by_model)
    plot_min_eigen_histograms(report_figures / "milestone4_min_eigen_histograms.png", records_by_model)
    plot_k3_quintic_baseline(report_figures / "milestone4_k3_quintic_fs_baseline.png", results)

    section = markdown_summary(results, theory_count=len(theory_rows))
    Path("reports/milestone4_section.md").write_text(section, encoding="utf-8")
    Path("reports/milestone4_log.md").write_text(
        "# Milestone 4 Run Log\n\n"
        f"Command: `uv run --extra dev python experiments/run_milestone4.py --max-points {args.max_points} --sample-count {args.sample_count} --ricci-points {args.ricci_points}`\n\n"
        + section,
        encoding="utf-8",
    )
    print(section)


if __name__ == "__main__":
    main()
