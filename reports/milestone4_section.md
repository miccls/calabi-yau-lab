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
