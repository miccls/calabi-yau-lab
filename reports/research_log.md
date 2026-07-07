# Research Log

## Initial Loop

- Created the canonical `calabi-yau-lab` repository structure and incorporated the first `cy_expansion_lab` research module.
- Added `MILESTONES.md`, `CHECKLIST.md`, `ARTIFACTS.md`, Git ignore rules, report files, configs, tests, and reproducible experiment commands.
- Milestone 0 verification commands: `uv run --extra dev pytest` and `uv run python experiments/run_all.py --config configs/fermat_quartic.yaml`.
- Implemented Milestone 1 local Fermat quartic K3 geometry diagnostics: affine patches, branch-stable hypersurface reconstruction, finite-difference complex Hessians, metric eigenvalue positivity, residue-form holomorphic volume density, and centered Monge-Ampere residuals.
- Milestone 1 verification commands: `uv run --extra dev pytest` passed with 9 tests and `uv run python experiments/run_all.py --config configs/fermat_quartic.yaml` regenerated validation with true local geometry diagnostics.
- Fubini-Study reference in the smoke run has positivity violation rate `0.0` and Monge-Ampere RMSE `0.39127958698215337` over `48` local patches.
- Implemented Milestone 2 symmetry-aware invariant K3 potential model `K = K_FS + sum_a theta_a B_a(r)`, trained against local Monge-Ampere residual and positivity penalty.
- Saved checkpoint: `artifacts/models/fermat_quartic_invariant_potential.json`.
- Trained model validation: `{'sampled_count': 48, 'used_count': 48, 'skipped_count': 0, 'max_patch_residual': 2.716545014592136e-15, 'metric_positivity': {'count': 48, 'min_eigenvalue': 0.13916119682219955, 'median_min_eigenvalue': 0.28847573814045074, 'mean_min_eigenvalue': 0.27693825896938634, 'positivity_violation_rate': 0.0}, 'monge_ampere': {'valid_count': 48, 'invalid_rate': 0.0, 'ma_residual_mean_abs': 0.03911521071407681, 'ma_residual_rmse': 0.0461676437850746, 'ma_residual_max_abs': 0.11988177446593601, 'ma_constant': 0.4901225777506147}}`.
- Implemented Fermat quartic/quintic configuration files.
- Implemented projective Fermat sampling with generic, coordinate-degenerate, equimodular, fixed-type, phase-orbit, and permutation-orbit samples.
- Implemented invariant features from normalized radii: power sums, elementary symmetric polynomials, and `I_k` placeholders.
- Implemented quotient-collapse, orbit-consistency, positivity proxy, and volume proxy diagnostics.
- Implemented constant baseline, log-product, and positive log-sum monomial models.
- Ran end-to-end experiment for `fermat_quartic`.

## Numerical Issues

- Finite-difference metric positivity and Monge-Ampere residuals remain available for validation, but Milestone 4 adds JAX autodiff benchmark diagnostics.
- Ricci scalar diagnostics are implemented as an opt-in nested-autodiff benchmark path and remain too expensive for large sweeps without further optimization.
- SOTA-scale model training still needs to move from finite-difference Hessians to autodiff-compatible training objectives.
- The current fitted scalar invariant models can fit the synthetic target while failing Kahler positivity on many local patches; future fitting must include geometric losses or positivity-aware parameterizations.
- The current positive log-sum model fixes exponents, so it tests coefficient fitting more than basis discovery.
- An initial fixed-type sampler perturbation moved points off the Fermat hypersurface; the sampler now reprojects by solving for the last coordinate after the perturbation.

## Surprising Observations

- Symmetry orbit consistency is essentially exact because the target and model inputs are invariant by construction. This is useful as a smoke test but not yet a sufficiency proof.

## Next Steps

- Milestone 5: train autodiff-compatible geometry-aware models on K3 and Fermat quintic using the Milestone 4 metric suite.

## Milestone 3 Run

- Ran SOTA-oriented sweep command and selected best model by held-out centered log-Monge-Ampere RMSE.
- Best model path: `artifacts/models/fermat_quartic_milestone3_best.json`.
- Best candidate: `{'candidate': 'rich_seed20260707_l20.0001', 'basis_set': 'rich', 'seed': 20260707, 'l2_weight': 0.0001, 'basis_size': 12, 'validation_centered_log_ma_rmse': 0.014675484785481115, 'validation_centered_log_ma_mae': 0.012640399376932007, 'validation_ma_max_abs': 0.029432253954787435, 'validation_pos_fail': 0.0, 'validation_min_eig': 0.14722139364631862, 'train_objective': 0.0003781106586043236}`.
- Generated figures under `reports/figures/`.
- Added metric-definition and literature-comparison tables with units/conventions.

# Milestone 4 Run Log

Command: `uv run --extra dev python experiments/run_milestone4.py --max-points 36 --sample-count 120 --ricci-points 1`

## Milestone 4: Autodiff Benchmark Infrastructure

This milestone adds a JAX autodiff geometry path and a metric suite that keeps training objectives separate from evaluation metrics.

Metric conventions are now machine-readable and distinguish centered log-Monge-Ampere residuals, sample-normalized volume-ratio errors, sigma-style L1 volume-ratio errors, positivity diagnostics, and opt-in Ricci scalar diagnostics.

The literature-theory registry currently contains 7 entries, including Mirjanic-Mishra Proposition 3.3 and Section 6 analytical properties of `phi`.

| target | model | centered log-MA RMSE | sigma L1 volume ratio | positivity violation | valid patches |
| --- | --- | ---: | ---: | ---: | ---: |
| fermat_quartic_k3 | fermat_quartic_k3_fubini_study | 0.346285 | 0.314426 | 0 | 36 |
| fermat_quartic_k3 | k3_milestone2_invariant | 0.0377692 | 0.0322206 | 0 | 36 |
| fermat_quartic_k3 | k3_milestone3_best | 0.0153594 | 0.0134178 | 0 | 36 |
| fermat_quintic_threefold | fermat_quintic_threefold_fubini_study | 0.70171 | 0.525657 | 0 | 36 |

Limitations: current sigma values are sample-average proxies, not proof of identical integration measure to any published table. Ricci scalar diagnostics are implemented as an opt-in autodiff path because nested Hessians are substantially more expensive than log-determinant metrics.
