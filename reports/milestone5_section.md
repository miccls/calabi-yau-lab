## Milestone 5: SOTA Numerical Training On K3 And Quintic

Milestone 5 adds autodiff-trained invariant Kähler-potential corrections for both the Fermat quartic K3 and Fermat quintic threefold. The training path precomputes local Hessian tensors with JAX autodiff and optimizes geometric losses over invariant correction coefficients.

Reported metrics use the Milestone 4 definitions. They are unweighted sample averages over valid positive local patches unless explicitly stated otherwise; sigma-style values remain sample-normalized volume-ratio proxies.

### Selected Models

| target | candidate | basis | loss | validation centered log-MA RMSE | validation sigma L1 | test centered log-MA RMSE | test sigma L1 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| fermat_quartic_k3 | k3_rich_log_ma_seed20260707 | rich | log_ma | 0.0138331 | 0.0113249 | 0.0149542 | 0.0113278 |
| fermat_quintic_threefold | quintic_rich_log_ma_seed20260708 | rich | log_ma | 0.108596 | 0.0805233 | 0.0636611 | 0.0498405 |

### Benchmark Comparison

| target | model | centered log-MA RMSE | sigma L1 volume ratio | positivity violation | valid patches |
| --- | --- | ---: | ---: | ---: | ---: |
| fermat_quartic_k3 | fermat_quartic_k3_fubini_study | 0.459023 | 0.420541 | 0 | 18 |
| fermat_quartic_k3 | k3_milestone2 | 0.0342019 | 0.0304118 | 0 | 18 |
| fermat_quartic_k3 | k3_milestone3 | 0.015023 | 0.010785 | 0 | 18 |
| fermat_quartic_k3 | k3_milestone5_best | 0.0149542 | 0.0113278 | 0 | 18 |
| fermat_quintic_threefold | fermat_quintic_threefold_fubini_study | 0.667783 | 0.564348 | 0 | 18 |
| fermat_quintic_threefold | quintic_milestone5_best | 0.0636611 | 0.0498405 | 0 | 18 |

### Figures

- [milestone5_training_curves.png](figures/milestone5_training_curves.png)
- [milestone5_loss_components.png](figures/milestone5_loss_components.png)
- [milestone5_centered_log_ma_bars.png](figures/milestone5_centered_log_ma_bars.png)
- [milestone5_sigma_l1_bars.png](figures/milestone5_sigma_l1_bars.png)
- [milestone5_residual_histograms.png](figures/milestone5_residual_histograms.png)
- [milestone5_volume_ratio_histograms.png](figures/milestone5_volume_ratio_histograms.png)
- [milestone5_min_eigen_histograms.png](figures/milestone5_min_eigen_histograms.png)
- [milestone5_stratum_residuals.png](figures/milestone5_stratum_residuals.png)
- [milestone5_model_size_vs_accuracy.png](figures/milestone5_model_size_vs_accuracy.png)
- [milestone5_basis_ablation.png](figures/milestone5_basis_ablation.png)
- [milestone5_loss_ablation.png](figures/milestone5_loss_ablation.png)
- [milestone5_sampling_counts.png](figures/milestone5_sampling_counts.png)
- [milestone5_k3_residual_vs_p2.png](figures/milestone5_k3_residual_vs_p2.png)
- [milestone5_quintic_residual_vs_p2.png](figures/milestone5_quintic_residual_vs_p2.png)

### Limitations

This milestone is the first autodiff-training pass, not a final SOTA claim. The implemented model family is an invariant linear correction in a widened feature basis, not yet a full neural or graph architecture. It is designed to produce reliable numerical targets, ablations, and failure-mode evidence for a Milestone 6 probing loop.
