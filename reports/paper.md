# Structured Invariant Expansions for Fermat Calabi-Yau Potentials

## Abstract

This experiment creates a quotient-first loop for discovering structured invariant expansions of Kahler potentials on Fermat-type Calabi-Yau hypersurfaces. The current target is `fermat_quartic`. Milestone 2 adds a saved symmetry-aware numerical potential model trained against local Monge-Ampere residuals.

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

The sampler creates generic points, near coordinate-degeneration points, near equimodular points, fixed-type points, and symmetry-related phase/permutation orbit copies. Outputs are stored in `data/experiments/fermat_quartic_smoke` with metadata and splits.

## Model Families

The loop compares a constant/Fubini-Study-like scalar baseline, a log-product invariant model, a positive log-sum generalized monomial model, and a trained K3 invariant potential of the form `K = K_FS + sum_a theta_a B_a(r)`.

## Optimization

The log-product model is linear least squares in log-features. The trained K3 invariant potential precomputes local Hessian tensors for each invariant basis term and optimizes coefficients against centered local Monge-Ampere residual with a metric-positivity penalty.

## Experiments and Results

| model | test_mse | test_mae | orbit_delta | min_eig | pos_fail | ma_rmse |
| --- | --- | --- | --- | --- | --- | --- |
| trained_invariant_potential | n/a | n/a | by construction | 0.139 | 0 | 0.0462 |
| constant_baseline | 0.037665 | 0.162392 | 0 | 0 | 1 | n/a |
| log_product | 0.000276783 | 0.0127398 | 2.45e-13 | -5.63e+07 | 0.854 | 0.964 |
| positive_log_sum_monomial | 0.000601347 | 0.0201694 | 4.1e-16 | -2.16 | 0.604 | 0.729 |

Fubini-Study local geometry reference:

```json
{
  "sampled_count": 48,
  "used_count": 48,
  "skipped_count": 0,
  "max_patch_residual": 2.716545014592136e-15,
  "metric_positivity": {
    "count": 48,
    "min_eigenvalue": 0.111551765789165,
    "median_min_eigenvalue": 0.34524758323300253,
    "mean_min_eigenvalue": 0.3096379743193846,
    "positivity_violation_rate": 0.0
  },
  "monge_ampere": {
    "valid_count": 48,
    "invalid_rate": 0.0,
    "ma_residual_mean_abs": 0.337389977090484,
    "ma_residual_rmse": 0.39127958698215337,
    "ma_residual_max_abs": 0.86261591706761,
    "ma_constant": 0.4885669669887678
  }
}
```

Trained invariant potential checkpoint:

```json
{
  "metadata": {
    "target": "fermat_quartic_k3",
    "basis_names": [
      "p2",
      "p3",
      "p4",
      "e2",
      "e3",
      "centered_l2"
    ],
    "seed": 20260707,
    "max_train_points": 36,
    "max_val_points": 48,
    "finite_difference_step": 0.0001,
    "positivity_weight": 10.0,
    "l2_weight": 0.001,
    "optimizer": "differential_evolution_then_nelder_mead",
    "train_objective": 0.0022041344980834256,
    "train_metrics": {
      "metric_positivity": {
        "count": 36,
        "min_eigenvalue": 0.16899048669803374,
        "median_min_eigenvalue": 0.29183874158690903,
        "mean_min_eigenvalue": 0.2867955548187575,
        "positivity_violation_rate": 0.0
      },
      "monge_ampere": {
        "valid_count": 36,
        "invalid_rate": 0.0,
        "ma_residual_mean_abs": 0.031189566483452797,
        "ma_residual_rmse": 0.038253469150199504,
        "ma_residual_max_abs": 0.10652589782703392,
        "ma_constant": 0.49772468859918184
      }
    },
    "validation_metrics": {
      "metric_positivity": {
        "count": 48,
        "min_eigenvalue": 0.15464378166707624,
        "median_min_eigenvalue": 0.28871245749391594,
        "mean_min_eigenvalue": 0.2840325918060459,
        "positivity_violation_rate": 0.0
      },
      "monge_ampere": {
        "valid_count": 48,
        "invalid_rate": 0.0,
        "ma_residual_mean_abs": 0.030371319195870394,
        "ma_residual_rmse": 0.03706743308856983,
        "ma_residual_max_abs": 0.06766832602504946,
        "ma_constant": 0.49616746777674026
      }
    },
    "baseline_train_metrics": {
      "metric_positivity": {
        "count": 36,
        "min_eigenvalue": 0.1361217395031727,
        "median_min_eigenvalue": 0.35497399065205165,
        "mean_min_eigenvalue": 0.31764819466653055,
        "positivity_violation_rate": 0.0
      },
      "monge_ampere": {
        "valid_count": 36,
        "invalid_rate": 0.0,
        "ma_residual_mean_abs": 0.2852255177098418,
        "ma_residual_rmse": 0.33399525835329247,
        "ma_residual_max_abs": 0.8452596079875399,
        "ma_constant": 0.4721608029172201
      }
    },
    "baseline_validation_metrics": {
      "metric_positivity": {
        "count": 48,
        "min_eigenvalue": 0.12422989130750171,
        "median_min_eigenvalue": 0.33255896191241163,
        "mean_min_eigenvalue": 0.3093489818297513,
        "positivity_violation_rate": 0.0
      },
      "monge_ampere": {
        "valid_count": 48,
        "invalid_rate": 0.0,
        "ma_residual_mean_abs": 0.3117145811106709,
        "ma_residual_rmse": 0.35124327383237636,
        "ma_residual_max_abs": 0.7024452368602045,
        "ma_constant": 0.45817857360845876
      }
    },
    "max_train_patch_residual": 2.1556631856153973e-15,
    "max_validation_patch_residual": 1.3911054626160788e-15,
    "config": "configs/fermat_quartic.yaml",
    "model_path": "artifacts/models/fermat_quartic_invariant_potential.json",
    "history_length": 632
  },
  "local_geometry": {
    "sampled_count": 48,
    "used_count": 48,
    "skipped_count": 0,
    "max_patch_residual": 2.716545014592136e-15,
    "metric_positivity": {
      "count": 48,
      "min_eigenvalue": 0.13916119682219955,
      "median_min_eigenvalue": 0.28847573814045074,
      "mean_min_eigenvalue": 0.27693825896938634,
      "positivity_violation_rate": 0.0
    },
    "monge_ampere": {
      "valid_count": 48,
      "invalid_rate": 0.0,
      "ma_residual_mean_abs": 0.03911521071407681,
      "ma_residual_rmse": 0.0461676437850746,
      "ma_residual_max_abs": 0.11988177446593601,
      "ma_constant": 0.4901225777506147
    }
  }
}
```

Quotient-collapse diagnostics:

```json
{
  "quotient_collapse": {
    "mean_abs_neighbor_delta": 0.004235624503189603,
    "median_abs_neighbor_delta": 0.002036513709513771,
    "max_abs_neighbor_delta": 0.06004370233180069
  },
  "target_orbit_consistency": {
    "mean_orbit_abs_delta": 9.561865068056292e-17,
    "max_orbit_abs_delta": 7.382983113757291e-15
  }
}
```

The best scalar model by held-out MSE is `log_product`.

## Plots and Tables

Scatter plots are written next to the experiment outputs:

- `data/experiments/fermat_quartic_smoke/constant_baseline_test_scatter.png`
- `data/experiments/fermat_quartic_smoke/log_product_test_scatter.png`
- `data/experiments/fermat_quartic_smoke/positive_log_sum_monomial_test_scatter.png`

## Discussion

The invariant quotient variables are sufficient for the synthetic target by construction, and the orbit-consistency diagnostics confirm that the fitted model outputs are nearly invariant under the generated symmetry orbits. The scalar log-product family is strong for the synthetic target, but it can fail Kahler positivity. The trained invariant potential is geometry-aware: it is optimized for local Monge-Ampere residual and metric positivity rather than scalar fit.

## Conclusions

Milestone 2 produces a reproducible saved K3 numerical potential checkpoint. It is still small and finite-difference trained, but it is the first model in the repo optimized directly against the true local geometry diagnostics. The next major need is SOTA-scale autodiff training on K3 and the Fermat quintic using the Milestone 4 metric suite.

## Limitations

This is not yet SOTA. The trained model is a low-dimensional invariant correction to Fubini-Study, optimized on a small local-patch subset with finite-difference Hessians. The local Monge-Ampere residual is patchwise and centered by fitting the additive volume constant over sampled points.

## Next Research Directions

1. Milestone 5: train autodiff-compatible geometry-aware models on K3 and Fermat quintic.
2. Run larger multi-seed and loss-ablation sweeps using the Milestone 4 metric suite.
3. Probe trained models on special loci and invariant coordinates.
4. Feed probing results back into training when they reveal better constraints, sampling, or architecture choices.
5. Use supported locus formulas and derivative bounds as symbolic-discovery constraints.


## Milestone 3: SOTA-Oriented K3 Tuning

Milestone 3 adds a reproducible sweep command:

```bash
uv run python experiments/run_milestone3.py --config configs/fermat_quartic_milestone3.yaml
```

The best selected model is saved at `artifacts/models/fermat_quartic_milestone3_best.json`. Selection is by held-out centered log-Monge-Ampere RMSE, with positivity violation rate used as the first ordering key.

Best candidate:

```json
{
  "candidate": "rich_seed20260707_l20.0001",
  "basis_set": "rich",
  "seed": 20260707,
  "l2_weight": 0.0001,
  "basis_size": 12,
  "validation_centered_log_ma_rmse": 0.014675484785481115,
  "validation_centered_log_ma_mae": 0.012640399376932007,
  "validation_ma_max_abs": 0.029432253954787435,
  "validation_pos_fail": 0.0,
  "validation_min_eig": 0.14722139364631862,
  "train_objective": 0.0003781106586043236
}
```

### Architecture And Symmetry

The tuned models keep the Kähler-potential form `K = K_FS + correction`. Corrections are linear combinations of invariant basis terms in normalized radii `r_i = |z_i|^2 / sum_j |z_j|^2`. This enforces projective scaling invariance, Fermat phase invariance, and coordinate permutation invariance by construction. The richer Milestone 3 basis includes higher power sums, elementary symmetric polynomials, centered moments, and entropy-like radius features.

### Training Objective

The trainer precomputes local Hessian tensors for each invariant basis term and optimizes coefficients against a held-out local geometry objective: centered log-Monge-Ampere residual plus positivity and L2 penalties. This is still finite-difference based; autodiff Hessians remain the main path toward SOTA-scale training.

### Metric Definitions And Units

| metric | formula | unit | aggregation |
| --- | --- | --- | --- |
| centered_log_ma_rmse | sqrt(mean((log det g - log |Omega|^2 - mean)^2)) | dimensionless log residual | RMSE over valid positive local patches |
| centered_log_ma_mae | mean(abs(log det g - log |Omega|^2 - mean)) | dimensionless log residual | MAE over valid positive local patches |
| positivity_violation_rate | mean(lambda_min(g) <= tolerance) | fraction of sampled local patches | mean over local patches |
| min_eigenvalue | min eigenvalue of Hermitian local metric matrix | local coordinate metric units | minimum over sampled local patches |

Do not compare these values directly to sigma-loss, Ricci scalar losses, raw volume-ratio errors, MSE, or MAE values from other papers unless the same convention and sampling measure are used.

### Repo Comparison

| model | split | centered_log_ma_rmse | centered_log_ma_mae | positivity_violation_rate | min_eigenvalue |
| --- | --- | --- | --- | --- | --- |
| Fubini-Study | test | 0.400489 | 0.354731 | 0 | 0.112122 |
| Milestone 2 | test | 0.0333682 | 0.0276708 | 0 | 0.138961 |
| Milestone 3 best | test | 0.0130716 | 0.0111775 | 0 | 0.142249 |
| constant_baseline | test | n/a | n/a | n/a | n/a |
| log_product | test | n/a | n/a | n/a | n/a |
| positive_log_sum_monomial | test | n/a | n/a | n/a | n/a |

### Literature Comparison

| source | model | metric | unit/convention | split | value | note |
| --- | --- | --- | --- | --- | --- | --- |
| this repo | Fubini-Study | centered_log_ma_rmse | dimensionless centered log residual; held-out local patches | test | 0.400489 | centered log-Monge-Ampere residual over held-out local patches |
| this repo | Milestone 2 | centered_log_ma_rmse | dimensionless centered log residual; held-out local patches | test | 0.0333682 | centered log-Monge-Ampere residual over held-out local patches |
| this repo | Milestone 3 best | centered_log_ma_rmse | dimensionless centered log residual; held-out local patches | test | 0.0130716 | centered log-Monge-Ampere residual over held-out local patches |
| this repo | constant_baseline | centered_log_ma_rmse | dimensionless centered log residual; held-out local patches | test | n/a | scalar synthetic-target fit only; geometry reported in run_all validation |
| this repo | log_product | centered_log_ma_rmse | dimensionless centered log residual; held-out local patches | test | n/a | scalar synthetic-target fit only; geometry reported in run_all validation |
| this repo | positive_log_sum_monomial | centered_log_ma_rmse | dimensionless centered log residual; held-out local patches | test | n/a | scalar synthetic-target fit only; geometry reported in run_all validation |
| CYJAX arXiv:2211.12520 | JAX algebraic/spectral Kahler-potential ansatz | paper/package accuracy metrics, not imported here | not directly normalized to this repo's centered log-MA RMSE | literature | not copied | Primary relevance is architecture: algebraic ansatz preserves Kahlerity and patch compatibility. |
| Fundamental-domain projections arXiv:2407.06914 | cymetric phi-model with non-trainable invariant canonicalization | cymetric-style losses/sigma measures in paper tables | not directly normalized to centered log-MA RMSE | literature | not copied | Primary relevance is invariantization; projection layers can change smoothness behavior. |
| Invariant/symbolic models arXiv:2412.19778 | extrinsic-symmetry-aware neural and symbolic approximations | Ricci curvature/scalar and model-specific losses | not directly normalized to centered log-MA RMSE | literature | not copied | Primary relevance is symmetry-aware hypothesis space and symbolic distillation. |
| Sharp Edges arXiv:2606.26892 | symmetry-aware/GNN Calabi-Yau metric pipeline | paper-specific Ricci/volume diagnostics | not directly normalized to centered log-MA RMSE | literature | not copied | Primary relevance is careful handling of symmetry, sampling, and failure modes. |

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
