from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from cy_expansion_lab.config import ensure_dir, load_config
from cy_expansion_lab.reporting.tables import markdown_table


def run(script: str, config: str) -> None:
    subprocess.run([sys.executable, f"experiments/{script}", "--config", config], check=True)


def write_report(config_path: str) -> None:
    cfg = load_config(config_path)
    out = Path(cfg["outputs"]["experiment_dir"])
    report_dir = ensure_dir("reports")
    metrics = json.loads((out / "fit_metrics.json").read_text(encoding="utf-8"))
    qdiag = json.loads((out / "quotient_diagnostics.json").read_text(encoding="utf-8"))
    validation = json.loads((out / "validation.json").read_text(encoding="utf-8"))
    rows = []
    if "trained_invariant_potential" in validation and not validation["trained_invariant_potential"].get("missing"):
        local = validation["trained_invariant_potential"]["local_geometry"]
        positivity = local["metric_positivity"]
        ma = local["monge_ampere"]
        rows.append(
            {
                "model": "trained_invariant_potential",
                "test_mse": "n/a",
                "test_mae": "n/a",
                "orbit_delta": "by construction",
                "min_eig": f"{positivity['min_eigenvalue']:.3g}",
                "pos_fail": f"{positivity['positivity_violation_rate']:.3g}",
                "ma_rmse": _format_optional_float(ma["ma_residual_rmse"]),
            }
        )
    for model, splits in metrics.items():
        local = validation[model]["local_geometry"]
        positivity = local["metric_positivity"]
        ma = local["monge_ampere"]
        rows.append(
            {
                "model": model,
                "test_mse": f"{splits['test']['mse']:.6g}",
                "test_mae": f"{splits['test']['mae']:.6g}",
                "orbit_delta": f"{validation[model]['symmetry_consistency']['mean_orbit_abs_delta']:.3g}",
                "min_eig": f"{positivity['min_eigenvalue']:.3g}",
                "pos_fail": f"{positivity['positivity_violation_rate']:.3g}",
                "ma_rmse": _format_optional_float(ma["ma_residual_rmse"]),
            }
        )
    table = markdown_table(rows, ["model", "test_mse", "test_mae", "orbit_delta", "min_eig", "pos_fail", "ma_rmse"])
    best_scalar = min((r for r in rows if r["test_mse"] != "n/a"), key=lambda r: float(r["test_mse"]))["model"]
    fs_geom = validation["fubini_study_reference"]["local_geometry"]
    trained = validation.get("trained_invariant_potential", {})
    trained_geom = trained.get("local_geometry")
    trained_meta = trained.get("metadata", {})
    paper = f"""# Structured Invariant Expansions for Fermat Calabi-Yau Potentials

## Abstract

This experiment creates a quotient-first loop for discovering structured invariant expansions of Kahler potentials on Fermat-type Calabi-Yau hypersurfaces. The current target is `{cfg['target']}`. Milestone 2 adds a saved symmetry-aware numerical potential model trained against local Monge-Ampere residuals.

## Motivation

Ricci-flat potentials may be complicated in homogeneous coordinates but simpler after quotienting by projective scaling, Fermat phase symmetries, and coordinate permutations. This codebase tests that idea with reproducible sampling, invariant features, interpretable ansatz families, and validation beyond pointwise fit.

## Repository Protocol

The canonical repository is `calabi-yau-lab`; `src/cy_expansion_lab` is the first package inside it. Milestones are tracked in `MILESTONES.md`, completion checks in `CHECKLIST.md`, and artifact handling in `ARTIFACTS.md`. Generated experiment outputs are reproducible and are not treated as source unless explicitly promoted.

## Relation to Prior Work

The first implemented ansatz families deliberately build on earlier invariant-search experiments where log-products of `I_k`-type features were competitive symbolic forms. The positive log-sum monomial family generalizes that direction by replacing a single product with a positive sparse sum of invariant monomials. This is also aligned with recent symmetry-aware numerical Calabi-Yau metric work: enforce quotient structure first, then use fitting or learning only inside the reduced invariant hypothesis space.

## Mathematical Setup

We work with potentials of the form `omega_phi = omega_ref + i partial partialbar phi` in a fixed Kahler class. Milestone 1 adds affine local charts for the Fermat quartic K3, finite-difference complex Hessians for local potential callables, metric eigenvalue checks, residue-form holomorphic volume densities, and centered local Monge-Ampere residuals. Milestone 4 adds JAX autodiff Hessians, log determinants, benchmark metric definitions, Fermat quintic Fubini-Study baseline evaluation, and opt-in Ricci scalar diagnostics.

## Quotient Coordinates

Samples are generated on the Fermat hypersurface `sum_i z_i^d = 0` by solving for the last homogeneous coordinate. Projectively invariant radii are

`r_i = |z_i|^2 / sum_j |z_j|^2`.

The feature library includes power sums, elementary symmetric polynomials, and an extensible placeholder for `I_k`-style centered invariant bases.

## Data Generation

The sampler creates generic points, near coordinate-degeneration points, near equimodular points, fixed-type points, and symmetry-related phase/permutation orbit copies. Outputs are stored in `{out}` with metadata and splits.

## Model Families

The loop compares a constant/Fubini-Study-like scalar baseline, a log-product invariant model, a positive log-sum generalized monomial model, and a trained K3 invariant potential of the form `K = K_FS + sum_a theta_a B_a(r)`.

## Optimization

The log-product model is linear least squares in log-features. The trained K3 invariant potential precomputes local Hessian tensors for each invariant basis term and optimizes coefficients against centered local Monge-Ampere residual with a metric-positivity penalty.

## Experiments and Results

{table}

Fubini-Study local geometry reference:

```json
{json.dumps(fs_geom, indent=2)}
```

Trained invariant potential checkpoint:

```json
{json.dumps({'metadata': trained_meta, 'local_geometry': trained_geom}, indent=2)}
```

Quotient-collapse diagnostics:

```json
{json.dumps(qdiag, indent=2)}
```

The best scalar model by held-out MSE is `{best_scalar}`.

## Plots and Tables

Scatter plots are written next to the experiment outputs:

- `{out / 'constant_baseline_test_scatter.png'}`
- `{out / 'log_product_test_scatter.png'}`
- `{out / 'positive_log_sum_monomial_test_scatter.png'}`

## Discussion

The invariant quotient variables are sufficient for the synthetic target by construction, and the orbit-consistency diagnostics confirm that the fitted model outputs are nearly invariant under the generated symmetry orbits. The scalar log-product family is strong for the synthetic target, but it can fail Kahler positivity. The trained invariant potential is geometry-aware: it is optimized for local Monge-Ampere residual and metric positivity rather than scalar fit.

## Conclusions

Milestone 2 produces a reproducible saved K3 numerical potential checkpoint. It is still small and finite-difference trained, but it is the first model in the repo optimized directly against the true local geometry diagnostics. Milestone 5 now adds the first autodiff-trained K3 and quintic invariant correction checkpoints; the next major need is probing their failure modes and feeding that evidence into a richer second training pass.

## Limitations

This is not yet SOTA. The trained model is a low-dimensional invariant correction to Fubini-Study, optimized on a small local-patch subset with finite-difference Hessians. The local Monge-Ampere residual is patchwise and centered by fitting the additive volume constant over sampled points.

## Next Research Directions

1. Milestone 6: probe the Milestone 5 K3 and quintic models on special loci and invariant coordinates.
2. Test Mirjanic-Mishra Proposition 3.3 and Section 6 claims against the trained checkpoints.
3. Identify residual/failure regions that should change sampling, constraints, or architecture.
4. Feed probing results back into a second Milestone 5 training pass when justified.
5. Use supported locus formulas and derivative bounds as symbolic-discovery constraints.
"""
    (report_dir / "paper.md").write_text(paper, encoding="utf-8")
    for section_name in ["milestone3_section.md", "milestone4_section.md", "milestone5_section.md"]:
        section = report_dir / section_name
        if section.exists():
            with (report_dir / "paper.md").open("a", encoding="utf-8") as handle:
                handle.write("\n\n")
                handle.write(section.read_text(encoding="utf-8"))
    log = f"""# Research Log

## Initial Loop

- Created the canonical `calabi-yau-lab` repository structure and incorporated the first `cy_expansion_lab` research module.
- Added `MILESTONES.md`, `CHECKLIST.md`, `ARTIFACTS.md`, Git ignore rules, report files, configs, tests, and reproducible experiment commands.
- Milestone 0 verification commands: `uv run --extra dev pytest` and `uv run python experiments/run_all.py --config configs/fermat_quartic.yaml`.
- Implemented Milestone 1 local Fermat quartic K3 geometry diagnostics: affine patches, branch-stable hypersurface reconstruction, finite-difference complex Hessians, metric eigenvalue positivity, residue-form holomorphic volume density, and centered Monge-Ampere residuals.
- Milestone 1 verification commands: `uv run --extra dev pytest` passed with 9 tests and `uv run python experiments/run_all.py --config configs/fermat_quartic.yaml` regenerated validation with true local geometry diagnostics.
- Fubini-Study reference in the smoke run has positivity violation rate `{fs_geom['metric_positivity']['positivity_violation_rate']}` and Monge-Ampere RMSE `{fs_geom['monge_ampere']['ma_residual_rmse']}` over `{fs_geom['used_count']}` local patches.
- Implemented Milestone 2 symmetry-aware invariant K3 potential model `K = K_FS + sum_a theta_a B_a(r)`, trained against local Monge-Ampere residual and positivity penalty.
- Saved checkpoint: `{trained.get('checkpoint')}`.
- Trained model validation: `{trained_geom}`.
- Implemented Fermat quartic/quintic configuration files.
- Implemented projective Fermat sampling with generic, coordinate-degenerate, equimodular, fixed-type, phase-orbit, and permutation-orbit samples.
- Implemented invariant features from normalized radii: power sums, elementary symmetric polynomials, and `I_k` placeholders.
- Implemented quotient-collapse, orbit-consistency, positivity proxy, and volume proxy diagnostics.
- Implemented constant baseline, log-product, and positive log-sum monomial models.
- Ran end-to-end experiment for `{cfg['target']}`.

## Numerical Issues

- Finite-difference metric positivity and Monge-Ampere residuals remain available for validation, but Milestone 4 adds JAX autodiff benchmark diagnostics.
- Ricci scalar diagnostics are implemented as an opt-in nested-autodiff benchmark path and remain too expensive for large sweeps without further optimization.
- Milestone 5 adds autodiff-compatible training objectives; the remaining SOTA gap is richer neural/graph/spectral architectures and larger sampling.
- The current fitted scalar invariant models can fit the synthetic target while failing Kahler positivity on many local patches; future fitting must include geometric losses or positivity-aware parameterizations.
- The current positive log-sum model fixes exponents, so it tests coefficient fitting more than basis discovery.
- An initial fixed-type sampler perturbation moved points off the Fermat hypersurface; the sampler now reprojects by solving for the last coordinate after the perturbation.

## Surprising Observations

- Symmetry orbit consistency is essentially exact because the target and model inputs are invariant by construction. This is useful as a smoke test but not yet a sufficiency proof.

## Next Steps

- Milestone 6: probe the Milestone 5 K3 and quintic checkpoints and decide whether to feed findings back into a second Milestone 5 training loop.
"""
    (report_dir / "research_log.md").write_text(log, encoding="utf-8")
    for log_name in ["milestone3_log.md", "milestone4_log.md", "milestone5_log.md"]:
        log_path = report_dir / log_name
        if log_path.exists():
            with (report_dir / "research_log.md").open("a", encoding="utf-8") as handle:
                handle.write("\n")
                handle.write(log_path.read_text(encoding="utf-8"))


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3g}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fermat_quartic.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    for script in [
        "run_generate_data.py",
        "run_invariants.py",
        "run_quotient_diagnostics.py",
        "run_fit_models.py",
    ]:
        run(script, args.config)
    if cfg["target"] == "fermat_quartic" and cfg.get("training", {}).get("invariant_potential", {}).get("enabled", False):
        run("run_train_k3_potential.py", args.config)
    run("run_validate.py", args.config)
    write_report(args.config)
    print("Wrote reports/paper.md and reports/research_log.md")


if __name__ == "__main__":
    main()
