## Milestone 5 Run

- Command: `uv run --extra dev python experiments/run_milestone5.py --sample-count 96 --train-points 18 --val-points 14 --test-points 18 --maxiter 30 --basis-sets compact,rich,sqrt_s2_baseline --losses log_ma,hybrid_sigma --seeds 20260707,20260708`.
- Ran autodiff-linear invariant correction sweeps for Fermat quartic K3 and Fermat quintic.
- Selected best models by validation positivity violation rate, centered log-Monge-Ampere RMSE, and sigma-style volume-ratio error.
- Best fermat_quartic_k3: `k3_rich_log_ma_seed20260707` at `artifacts/models/fermat_quartic_milestone5_best.json`; test centered log-MA RMSE `0.014954184943359919`, test sigma-style L1 `0.011327841913471784`.
- Best fermat_quintic_threefold: `quintic_rich_log_ma_seed20260708` at `artifacts/models/fermat_quintic_milestone5_best.json`; test centered log-MA RMSE `0.06366106870454061`, test sigma-style L1 `0.049840509667919274`.
- Generated Milestone 5 figures under `reports/figures/milestone5_*.png` and tables under `reports/tables/milestone5_*.csv`.
- Published comparisons remain convention-aware; no external numeric table values were copied into repo comparisons.
