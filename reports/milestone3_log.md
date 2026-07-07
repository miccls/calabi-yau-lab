## Milestone 3 Run

- Ran SOTA-oriented sweep command and selected best model by held-out centered log-Monge-Ampere RMSE.
- Best model path: `artifacts/models/fermat_quartic_milestone3_best.json`.
- Best candidate: `{'candidate': 'rich_seed20260707_l20.0001', 'basis_set': 'rich', 'seed': 20260707, 'l2_weight': 0.0001, 'basis_size': 12, 'validation_centered_log_ma_rmse': 0.014675484785481115, 'validation_centered_log_ma_mae': 0.012640399376932007, 'validation_ma_max_abs': 0.029432253954787435, 'validation_pos_fail': 0.0, 'validation_min_eig': 0.14722139364631862, 'train_objective': 0.0003781106586043236}`.
- Generated figures under `reports/figures/`.
- Added metric-definition and literature-comparison tables with units/conventions.
