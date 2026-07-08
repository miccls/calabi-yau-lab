# Calabi-Yau Lab

Research codebase for quotient-first discovery of structured invariant expansions of Calabi-Yau Kahler potentials.

The canonical local repository is `/Users/ms/calabi-yau-lab`. The Python package currently lives under `src/cy_expansion_lab` because the first research module is the invariant-expansion lab.

The current implementation is a runnable Fermat quartic K3 and Fermat quintic benchmark loop. It samples Fermat-type hypersurfaces, constructs projectively invariant quotient coordinates, fits interpretable invariant models, computes local Kähler geometry diagnostics, trains a small symmetry-aware numerical potential model, evaluates JAX autodiff geometry metrics, and writes research reports.

## Setup

```bash
cd /Users/ms/calabi-yau-lab
uv run --extra dev pytest
```

## First Experiment

```bash
cd /Users/ms/calabi-yau-lab
uv run python experiments/run_all.py --config configs/fermat_quartic.yaml
```

Outputs are written to:

- `data/experiments/fermat_quartic_smoke/`
- `artifacts/models/fermat_quartic_invariant_potential.json`
- `artifacts/models/fermat_quartic_milestone3_best.json`
- `reports/paper.md`
- `reports/research_log.md`

## Milestone 3 Tuning

```bash
cd /Users/ms/calabi-yau-lab
uv run python experiments/run_milestone3.py --config configs/fermat_quartic_milestone3.yaml
```

This runs a small SOTA-oriented invariant-basis sweep, selects the best model by held-out centered log-Monge-Ampere RMSE with positivity violation rate as the first ordering key, saves the best checkpoint, and generates figures under `reports/figures/`.

## Milestone 4 Autodiff Benchmark

```bash
cd /Users/ms/calabi-yau-lab
uv run --extra dev python experiments/run_milestone4.py --max-points 36 --sample-count 120 --ricci-points 1
```

This runs the JAX autodiff evaluation path for:

- Fermat quartic K3 Fubini-Study baseline.
- Fermat quartic K3 Milestone 2 checkpoint.
- Fermat quartic K3 Milestone 3 checkpoint.
- Fermat quintic threefold Fubini-Study baseline.

Tracked summaries are written to `reports/tables/`, `reports/milestone4_section.md`, and `reports/milestone4_log.md`. Generated sample-level outputs are written to `data/experiments/milestone4_benchmark/`.

## Milestone 5 Autodiff Training

```bash
cd /Users/ms/calabi-yau-lab
uv run --extra dev python experiments/run_milestone5.py --sample-count 96 --train-points 18 --val-points 14 --test-points 18 --maxiter 30 --basis-sets compact,rich,sqrt_s2_baseline --losses log_ma,hybrid_sigma --seeds 20260707,20260708
```

This runs the first SOTA-oriented autodiff training sweep for:

- Fermat quartic K3 invariant Kähler-potential corrections.
- Fermat quintic invariant Kähler-potential corrections.
- Compact, rich, and `sqrt(s2)`-style theory-derived invariant bases.
- Centered log-Monge-Ampere and hybrid sigma/volume-ratio losses.

Tracked outputs are written to:

- `artifacts/models/fermat_quartic_milestone5_best.json`
- `artifacts/models/fermat_quintic_milestone5_best.json`
- `reports/tables/milestone5_*.csv`
- `reports/figures/milestone5_*.png`
- `reports/milestone5_section.md`
- `reports/milestone5_log.md`

## Research Protocol

- `MILESTONES.md` defines the sequential research plan.
- `CHECKLIST.md` defines the required completion checks for each milestone.
- `ARTIFACTS.md` defines what should be committed and what should remain generated or externally stored.
- `reports/research_log.md` records implementation attempts, failures, numerical issues, and next steps.
- `reports/paper.md` and `reports/paper.tex` track the research story as results become meaningful.

## What Is Implemented

- Fermat quartic K3 and Fermat quintic configuration files.
- Reproducible projective sampling on `sum_i z_i^d = 0`.
- Stratified samples: generic, coordinate-degenerate, equimodular, fixed-type, phase orbits, and permutation orbits.
- Invariant coordinates from normalized radii `r_i = |z_i|^2 / sum_j |z_j|^2`.
- Power sums, elementary symmetric polynomials, and an extensible `I_k` placeholder.
- Quotient-collapse and symmetry-orbit diagnostics.
- Constant baseline, log-product model, and positive log-sum monomial model.
- Affine local Fermat quartic K3 patches.
- Finite-difference complex Hessian diagnostics.
- JAX autodiff complex Hessian diagnostics.
- Metric eigenvalue positivity checks.
- Residue-form holomorphic volume density.
- Centered local Monge-Ampere residuals.
- Sample-normalized raw volume-ratio and sigma-style L1 volume-ratio metrics.
- Opt-in autodiff Ricci scalar diagnostics.
- Symmetry-aware numerical potential model `K = K_FS + sum_a theta_a B_a(r)`.
- Saved K3 checkpoint at `artifacts/models/fermat_quartic_invariant_potential.json`.
- Milestone 3 tuned checkpoint at `artifacts/models/fermat_quartic_milestone3_best.json`.
- Milestone 5 autodiff-trained K3 checkpoint at `artifacts/models/fermat_quartic_milestone5_best.json`.
- Milestone 5 autodiff-trained quintic checkpoint at `artifacts/models/fermat_quintic_milestone5_best.json`.
- Fermat quintic Fubini-Study autodiff benchmark baseline.
- Autodiff-linear invariant correction training for both K3 and quintic.
- Basis/loss/seed sweeps with held-out metric selection.
- Literature-theory registry for imported propositions/conjectures, including Mirjanic-Mishra Proposition 3.3 and Section 6 analytical properties of `phi`.
- Metric-definition and comparison tables with explicit units/conventions.
- Figures for training curves, residual distributions, stratum-wise residuals, positivity, correction structure, ablations, volume-ratio metrics, and baseline comparisons.

## Current Approximation Boundary

This project does not yet claim a closed-form Ricci-flat metric. Milestone 5 adds the first autodiff-trained K3 and quintic invariant correction models, but the model family is still a linear invariant correction, not a full neural, graph, or high-dimensional spectral SOTA architecture. Sigma-style values are sample-average proxies unless a future experiment matches the sampling measure and normalization of a specific published table.

## Project Layout

```text
configs/                 experiment configs
data/                    generated samples and experiment outputs
experiments/             runnable scripts
artifacts/               promoted checkpoints, models, or external artifacts
reports/                 generated report and research log
src/cy_expansion_lab/    package source
tests/                   smoke and unit tests
```

## Research Direction

The working hypothesis is that Ricci-flat Kahler potentials may become sparse or structured after quotienting by projective scaling, Fermat phase symmetries, and coordinate permutations. The next serious milestones are:

1. Probe the Milestone 5 K3 and quintic models on special loci and invariant coordinates.
2. Test Mirjanic-Mishra Proposition 3.3 and Section 6 claims against the trained models.
3. Identify residual/failure regions that should change sampling, constraints, or architecture.
4. Feed Milestone 6 probing results back into a second Milestone 5 training pass when justified.
5. Use supported locus formulas and derivative bounds as symbolic-discovery constraints.
