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
