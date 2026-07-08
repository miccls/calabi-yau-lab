# Calabi-Yau Lab Milestones

This file tracks the sequential research program for moving from the current experiment scaffold toward structured analytical expansions, or closed-form candidates, for Ricci-flat Kahler potentials on highly symmetric Calabi-Yau manifolds.

The near-term focus is the Fermat quartic K3. The quintic threefold should be treated as the first major generalization target once the K3 pipeline is geometrically reliable.

## Guiding Principle

Symbolic discovery should be driven by extremely accurate numerical geometry. The project should first build a trustworthy, symmetry-aware numerical model of the Ricci-flat K3 potential, then probe that model for invariant structure, then use constrained symbolic search to identify candidate expansions.

## Milestone Status

- [x] Milestone 0: Repository And Research Protocol
- [x] Milestone 1: Exact Local K3 Geometry Infrastructure
- [x] Milestone 2: High-Quality Symmetry-Aware Numerical Potential Model
- [x] Milestone 3: SOTA K3 Numerical Tuning
- [x] Milestone 4: Benchmark-Grade Autodiff Geometry And Published Loss Suite
- [x] Milestone 5: SOTA Numerical Training On K3 And Quintic
- [ ] Milestone 6: Numerical Model Probing And Failure-Mode Analysis
- [ ] Milestone 7: Quotient Sufficiency And Invariant Basis Discovery
- [ ] Milestone 8: Structured Symbolic Ansatz Search
- [ ] Milestone 9: Locus-Constrained Symbolic Discovery
- [ ] Milestone 10: Candidate Formula Evaluation
- [ ] Milestone 11: Cross-Geometry Generalization
- [ ] Milestone 12: Analytical Interpretation

## Milestone 0: Repository And Research Protocol

Create the canonical research workspace and make every future experiment reproducible.

Deliverables:

- Clean repository structure, eventually under `calabi-yau-lab`.
- `README.md` with setup and first experiment commands.
- `MILESTONES.md`.
- `CHECKLIST.md` or the checklist in this file.
- `reports/research_log.md`.
- `reports/paper.md` and/or `reports/paper.tex`.
- Reproducible configs for Fermat quartic K3 and Fermat quintic.
- Smoke tests for the basic data, invariant, fitting, and validation loop.
- A clear policy for saved models, generated data, and long-running artifacts.

Completion criteria:

- The repository can be set up from scratch with documented commands.
- At least one minimal experiment runs end to end.
- Reports and logs are updated from actual outputs.

Completion record:

- Canonical repo: `/Users/ms/calabi-yau-lab`.
- Existing `cy_expansion_lab` work incorporated as `src/cy_expansion_lab`.
- Local Git repo initialized.
- GitHub-ready `.gitignore` and CI workflow added.
- Artifact policy added in `ARTIFACTS.md`.
- Completion checklist added in `CHECKLIST.md`.
- Verified command: `uv run --extra dev pytest` passed with 4 tests.
- Verified command: `uv run python experiments/run_all.py --config configs/fermat_quartic.yaml` completed and regenerated the first report/log.
- Next suggested milestone: Milestone 1, exact local K3 geometry infrastructure.

## Milestone 1: Exact Local K3 Geometry Infrastructure

Replace proxy diagnostics with true local-patch Kahler geometry for the Fermat quartic K3.

Deliverables:

- Affine patch coordinates for the Fermat quartic.
- Stable handling of the hypersurface constraint.
- Pullback of ambient/projective potentials to local coordinates.
- Complex Hessian computation:

  ```text
  g_{i jbar} = partial_i partial_jbar K
  ```

- Metric eigenvalue and positivity checks.
- Holomorphic volume form in local patches.
- Monge-Ampere residual:

  ```text
  log det(g) - log(Omega wedge Omegabar) - constant
  ```

- Patch transition consistency tests.
- Integration into `experiments/run_validate.py`.

Completion criteria:

- The quartic experiment reports true metric positivity and true Monge-Ampere residuals, not only proxies.
- Tests cover local coordinates, Hessians, positivity checks, and patch consistency.

Completion record:

- Implemented affine local patches for the Fermat quartic K3 in `src/cy_expansion_lab/cy/local_patch.py`.
- Each patch fixes a projective gauge, eliminates a stable coordinate using the hypersurface equation, and keeps a branch factor matched to the base sample.
- Implemented finite-difference complex Hessians for local potential callables.
- Implemented Fubini-Study patch metric diagnostics.
- Implemented residue-form holomorphic volume density `|Omega|^2 = |dP/dz_elim|^{-2}` in local coordinates.
- Implemented Hermitian metric eigenvalue statistics and positivity violation rates.
- Implemented centered local Monge-Ampere residual diagnostics in `src/cy_expansion_lab/validate/monge_ampere.py`.
- Wired the diagnostics into `experiments/run_validate.py` and `experiments/run_all.py`.
- Added tests in `tests/test_local_geometry.py`.
- Verified command: `uv run --extra dev pytest` passed with 9 tests.
- Verified command: `uv run python experiments/run_all.py --config configs/fermat_quartic.yaml` completed and reported true local geometry diagnostics.
- Fubini-Study reference on 48 samples: positivity violation rate `0.0`, Monge-Ampere RMSE `0.39127958698215337`.
- Current fitted invariant models are not reliable Kahler potentials: log-product positivity violation rate `0.8541666666666666`, positive log-sum positivity violation rate `0.6041666666666666`.
- Remaining limitation: Hessians are finite-difference validation tools; autodiff Hessians are still needed for training-scale optimization.
- Remaining limitation: Ricci tensor residuals are not implemented.
- Next suggested milestone: Milestone 2, high-quality symmetry-aware numerical potential model.

## Milestone 2: High-Quality Symmetry-Aware Numerical Potential Model

Implement a serious numerical model for the Ricci-flat Kahler potential on the Fermat quartic K3.

Deliverables:

- A symmetry-aware potential model, such as a neural invariant model, Bergman model, or hybrid.
- The model must preserve the Kahler-potential formulation rather than directly fitting arbitrary Hermitian metrics.
- Projective, Fermat phase, and permutation symmetries enforced by construction where feasible.
- Gauge fixing, such as mean-zero normalization or fixed value at a reference point.
- Training against true Monge-Ampere loss.
- Validation on held-out strata.
- Optional hard constraints on known loci.
- Saved model checkpoints and configs in a permanent artifact path.

Completion criteria:

- A trained K3 numerical potential model exists and can be loaded reproducibly.
- The model beats simple Fubini-Study/reference baselines on geometric residuals.
- The saved checkpoint is documented in the research log and report.

Completion record:

- Implemented `InvariantPotentialModel`, a symmetry-aware Kähler potential ansatz `K = K_FS + sum_a theta_a B_a(r)` where `B_a` are projective, Fermat-phase, and permutation invariant basis terms.
- Implemented local Hessian-tensor precomputation and geometric coefficient fitting in `src/cy_expansion_lab/fit/k3_potential.py`.
- Training objective uses centered local Monge-Ampere residual plus a metric-positivity penalty.
- Added training command `experiments/run_train_k3_potential.py`.
- Wired training into the standard quartic `experiments/run_all.py` flow.
- Saved checkpoint: `artifacts/models/fermat_quartic_invariant_potential.json`.
- Verified command: `uv run --extra dev pytest` passed with 12 tests.
- Verified command: `uv run python experiments/run_all.py --config configs/fermat_quartic.yaml` completed, trained the model, saved the checkpoint, and regenerated validation/report artifacts.
- Training metadata validation subset: Fubini-Study baseline Monge-Ampere RMSE `0.35124327383237636`; trained model Monge-Ampere RMSE `0.03706743308856983`; positivity violation rate `0.0`.
- Full smoke validation block: Fubini-Study reference Monge-Ampere RMSE `0.39127958698215337`; trained model Monge-Ampere RMSE `0.0461676437850746`; positivity violation rate `0.0`.
- Remaining limitation: this is a small finite-difference-trained invariant correction model, not SOTA.
- Remaining limitation: no autodiff Hessians, neural invariant architecture, multi-seed tuning, or locus constraints yet.
- Next suggested milestone: Milestone 3, SOTA K3 numerical tuning.

## Milestone 3: SOTA K3 Numerical Tuning

Push the numerical K3 potential model toward state-of-the-art accuracy.

Please consider current SOTA methods when designing the models.
If you copy something from another SOTA implementation, refer to which.
If something is completely new, state that too.

References:

* CYJAX: https://arxiv.org/abs/2211.12520
    - Known drawbacks: Scales poorly as the spectral basis grows very quickly

* Fundamental domain projections: https://arxiv.org/abs/2407.06914
    - Known drawbacks: The projections may not respect the smooth behavior of the analytical potential as they may introduce kinks in the Ricci measure. May achieve good values of the loss measures but due the f$

* Invariant networks: https://arxiv.org/abs/2412.19778, https://arxiv.org/pdf/2606.26892
    - Looks promising, hopefully something we can build on.
    - Can we use anything from different fields of ML which these authors miss? Are there any underlying structure which fit properly into existing methods that are not leveraged in these papers?

Deliverables:

- Multi-seed training runs.
- Hyperparameter sweeps.
- Improved stratified sampling near difficult strata.
- Locus-constrained training if reliable locus formulas are available.
- Symmetry orbit consistency checks.
- Residual heatmaps and stratum-specific validation tables.
- Comparison against literature metrics where available.
- Permanent best checkpoint and exact command/config for regeneration.

Completion criteria:

- Best model is selected by held-out geometric metrics.
- Failure modes are mapped by stratum.
- The project has a stable numerical reference model accurate enough to mine for symbolic structure.

Completion record:

- Implemented Milestone 3 sweep command: `uv run python experiments/run_milestone3.py --config configs/fermat_quartic_milestone3.yaml`.
- Added richer invariant basis sets with higher power sums, elementary symmetric polynomials, centered moments, and entropy-like radius features.
- Added multi-seed and multi-regularization sweeps.
- Best model selected by held-out centered log-Monge-Ampere RMSE with positivity violation rate as the first ordering key.
- Best checkpoint: `artifacts/models/fermat_quartic_milestone3_best.json`.
- Best candidate: `rich_seed20260707_l20.0001`.
- Held-out test centered log-Monge-Ampere RMSE: Fubini-Study `0.4004889293486422`, Milestone 2 `0.033368181435130775`, Milestone 3 best `0.013071553239515454`.
- Positivity violation rate for best model: `0.0`.
- Generated figures under `reports/figures/`: training curves, minimum eigenvalue histogram, residual histogram, stratum residuals, correction-vs-invariant plot, and comparison bar chart.
- Added metric-definition table and literature comparison table with explicit units/conventions.
- External literature is compared by method and metric convention, not by copied numeric values, because CYJAX, fundamental-domain projection, invariant-network, and GNN/symmetry papers use different loss conventions.
- Verified command: `uv run --extra dev pytest` passed with 16 tests.
- Verified command: `uv run python experiments/run_milestone3.py --config configs/fermat_quartic_milestone3.yaml` completed.
- Remaining limitation: finite-difference Hessians still limit training scale; autodiff/JAX/PyTorch Hessians remain needed.
- Remaining limitation: Ricci residuals are still not implemented.
- Next suggested milestone: Milestone 4, benchmark-grade autodiff geometry and published loss suite.

## Milestone 4: Benchmark-Grade Autodiff Geometry And Published Loss Suite

Replace finite-difference geometry with an autodiff-capable evaluation path and implement the loss measures needed for fair comparison with published Calabi-Yau metric-learning work.

Deliverables:

- Autodiff backend, preferably JAX unless the repo structure strongly favors another backend.
- Autodiff computation of Kahler-potential derivatives, Hermitian metric components, determinant/log determinant, Monge-Ampere residuals, and positivity diagnostics.
- Ricci tensor and scalar-curvature diagnostics where computationally practical.
- Published metric/loss suite with exact formulas, units, and aggregation conventions:
  - centered log-Monge-Ampere RMSE/MAE;
  - raw Monge-Ampere volume-ratio error;
  - sigma measure / sigma loss, matching literature convention as closely as possible;
  - Ricci tensor/scalar diagnostics;
  - Kahler-condition diagnostics where relevant;
  - transition/patch consistency diagnostics where relevant.
- Separation between training objectives and evaluation metrics.
- Fermat quintic threefold support for baseline evaluation:
  - point sampling;
  - local patch reconstruction;
  - holomorphic volume density;
  - Fubini-Study baseline;
  - model evaluation.
- Benchmark runner comparing at least Fubini-Study, the current K3 checkpoints, and any autodiff-compatible model.
- Metric-definition tables in machine-readable and report form.
- Literature-theory registry for exact propositions, conjectures, and usable constraints from key papers.
- Initial entries for Mirjanic-Mishra 2025, especially Proposition 3.3 and Section 6 analytical properties of `phi`:
  - phase-independence criterion for coordinates appearing only as Fermat monomials;
  - cross-dimensional self-similarity conjecture for `phi` as a function of invariant coordinates;
  - derivative-size inequalities for `phi` and its first/second derivatives;
  - pseudo-origin and equimodular-locus metric formulae;
  - near-equimodular approximation using a scaled Fubini-Study pullback.

Completion criteria:

- Autodiff Hessian/log-determinant path exists and is tested against finite differences on small K3 samples.
- Every reported metric states whether it is MSE, RMSE, MAE, L1, L2, volume-ratio, log-volume residual, Ricci scalar, or tensor norm.
- Metric normalization is explicit: sample average, volume-weighted average, dimension-dependent constants, centering constants, and whether absolute value/square/root is used.
- K3 experiments still reproduce the Milestone 3 model evaluation or document any numerical convention change clearly.
- Fermat quintic Fubini-Study baseline evaluation runs successfully.
- Benchmark reports and figures are generated and reproducible.
- The paper-derived theory registry records each imported claim with source, formula, implementation status, and whether it is proven, conjectural, or empirical.

Completion record:

- Added JAX as the autodiff backend.
- Implemented branch-stable JAX local Fermat patch reconstruction, Fubini-Study patch potentials, invariant-model patch potentials, complex Hessians, log determinants, and opt-in Ricci scalar diagnostics.
- Added benchmark metric definitions for centered log-Monge-Ampere RMSE/MAE, sample-normalized raw volume-ratio errors, sigma-style L1 volume-ratio error, positivity diagnostics, and Ricci scalar diagnostics.
- Added Fermat quintic Fubini-Study baseline evaluation through the same autodiff metric suite.
- Added a literature-theory registry with entries for Mirjanic-Mishra Proposition 3.3 and Section 6 analytical properties of `phi`.
- Added benchmark command: `uv run --extra dev python experiments/run_milestone4.py --max-points 36 --sample-count 120 --ricci-points 1`.
- Milestone 4 autodiff benchmark results:
  - Fermat quartic K3 Fubini-Study: centered log-MA RMSE `0.346285`, sigma-style L1 volume-ratio error `0.314426`, positivity violation `0.0`.
  - Fermat quartic K3 Milestone 2 invariant checkpoint: centered log-MA RMSE `0.0377692`, sigma-style L1 volume-ratio error `0.0322206`, positivity violation `0.0`.
  - Fermat quartic K3 Milestone 3 best checkpoint: centered log-MA RMSE `0.0153594`, sigma-style L1 volume-ratio error `0.0134178`, positivity violation `0.0`.
  - Fermat quintic threefold Fubini-Study: centered log-MA RMSE `0.70171`, sigma-style L1 volume-ratio error `0.525657`, positivity violation `0.0`.
- Generated tracked figures under `reports/figures/milestone4_*.png`.
- Generated tracked metric and theory tables under `reports/tables/`.
- Generated run section and log under `reports/milestone4_section.md` and `reports/milestone4_log.md`.
- Verified command: `uv run --extra dev pytest` passed with 20 tests.
- Remaining limitation: sigma-style values are sample-average proxies, not proof of identical sampling measure or normalization to any published table.
- Remaining limitation: Ricci scalar diagnostics are implemented but intentionally opt-in because nested autodiff Hessians are substantially more expensive.
- Next suggested milestone: Milestone 5, SOTA numerical training on K3 and quintic.

## Milestone 5: SOTA Numerical Training On K3 And Quintic

Train genuinely competitive numerical models using the benchmark-grade autodiff and metric infrastructure.

Deliverables:

- Autodiff-trained K3 model that improves over the Milestone 3 finite-difference invariant model.
- Fermat quintic training run using the same metric definitions as the benchmark suite.
- Geometry-aware and symmetry-aware model families, including invariant networks, algebraic ansatz variants, or hybrid architectures.
- Stratified and difficult-locus sampling strategy.
- Optional hard or soft locus constraints when reliable formulas are available.
- Constraint/loss ablations using the Mirjanic-Mishra analytical properties where appropriate:
  - phase-invariance features from Proposition 3.3;
  - derivative-bound regularizers inspired by Conjecture 6.3;
  - pseudo-origin metric targets from Proposition 6.4;
  - equimodular metric targets from Proposition 6.5 when the symmetry assumptions are accepted for that experiment;
  - near-equimodular scaled-Fubini-Study warm starts or auxiliary losses from Conjecture 6.6;
  - simple `sqrt(s2)` correction baseline from Proposition 6.7.
- Multi-seed, hyperparameter, architecture, and loss-ablation sweeps.
- Permanent best checkpoints with configs, seeds, exact commands, and metric metadata.
- Publication-quality figures:
  - training curves;
  - evaluation metric comparison bars;
  - residual histograms;
  - residuals by stratum/locus;
  - K3 versus quintic comparisons where meaningful.
- Literature comparison table using compatible metric definitions only.

Completion criteria:

- Best models are selected by held-out benchmark metrics, not training loss alone.
- K3 and quintic results are reported with the same metric-definition table.
- The report states which losses were optimized and which metrics were evaluation-only.
- Any theory-derived constraint is reported as hard constraint, soft penalty, feature restriction, warm start, or evaluation-only diagnostic.
- The best checkpoints are saved permanently and documented.
- Remaining gap to published work is stated in comparable units where possible.

Completion record:

- Implemented generic autodiff-linear invariant correction training in `src/cy_expansion_lab/fit/autodiff_linear.py`.
- Extended invariant basis support with `centered_l5`, `centered_l6`, and a smoothed `sqrt_centered_l2` term to evaluate the Mirjanic-Mishra `sqrt(s2)`-style baseline.
- Added Milestone 5 runner: `experiments/run_milestone5.py`.
- Ran the main Milestone 5 command:

  ```bash
  uv run --extra dev python experiments/run_milestone5.py --sample-count 96 --train-points 18 --val-points 14 --test-points 18 --maxiter 30 --basis-sets compact,rich,sqrt_s2_baseline --losses log_ma,hybrid_sigma --seeds 20260707,20260708
  ```

- Implemented basis, loss, and seed sweeps for Fermat quartic K3 and Fermat quintic.
- Implemented training losses based on centered log-Monge-Ampere MSE, sample-normalized volume-ratio MSE, sigma-style smooth L1 volume-ratio error, positivity penalty, and L2 regularization.
- Saved best K3 checkpoint: `artifacts/models/fermat_quartic_milestone5_best.json`.
- Saved best quintic checkpoint: `artifacts/models/fermat_quintic_milestone5_best.json`.
- Best K3 candidate: `k3_rich_log_ma_seed20260707`.
- Best quintic candidate: `quintic_rich_log_ma_seed20260708`.
- Autodiff benchmark comparison on the Milestone 5 held-out test patches:
  - Fermat quartic K3 Fubini-Study: centered log-MA RMSE `0.459023`, sigma-style L1 volume-ratio error `0.420541`, positivity violation `0.0`.
  - Fermat quartic K3 Milestone 2 checkpoint: centered log-MA RMSE `0.0342019`, sigma-style L1 volume-ratio error `0.0304118`, positivity violation `0.0`.
  - Fermat quartic K3 Milestone 3 checkpoint: centered log-MA RMSE `0.015023`, sigma-style L1 volume-ratio error `0.010785`, positivity violation `0.0`.
  - Fermat quartic K3 Milestone 5 best: centered log-MA RMSE `0.0149542`, sigma-style L1 volume-ratio error `0.0113278`, positivity violation `0.0`.
  - Fermat quintic Fubini-Study: centered log-MA RMSE `0.667783`, sigma-style L1 volume-ratio error `0.564348`, positivity violation `0.0`.
  - Fermat quintic Milestone 5 best: centered log-MA RMSE `0.0636611`, sigma-style L1 volume-ratio error `0.0498405`, positivity violation `0.0`.
- Generated 14 promoted figures under `reports/figures/milestone5_*.png`, including training curves, loss components, benchmark bars, residual histograms, volume-ratio histograms, positivity histograms, stratum residuals, ablation plots, sampling counts, and residual-vs-`p2` plots.
- Generated promoted tables under `reports/tables/milestone5_*.csv`.
- Added literature refresh table with CYJAX, CYJAX docs, Mirjanic-Mishra, symbolic distillation, Sharp Edges, and the local thesis extraction note.
- Theory-derived constraints are currently feature restrictions, basis terms, and evaluation-only diagnostics. Pseudo-origin/equimodular hard target losses and derivative-bound regularizers are explicitly deferred to Milestone 6 because compatible coordinate/locus formula implementations are not yet validated.
- Remaining limitation: this is an autodiff-trained invariant linear correction family, not yet a full neural, graph, or high-dimensional spectral SOTA architecture.
- Remaining limitation: K3 improves over Milestone 3 on centered log-MA RMSE by a small amount on the Milestone 5 held-out test set, but Milestone 3 remains slightly better on sigma-style L1 volume-ratio error.
- Next suggested milestone: Milestone 6, numerical model probing and failure-mode analysis, with a likely feedback loop into a second Milestone 5 iteration.

## Milestone 6: Numerical Model Probing And Failure-Mode Analysis

Use the best trained numerical models as experimental objects and feed discoveries back into Milestone 5 when they suggest better architectures, losses, constraints, or sampling.

Deliverables:

- Behavior on equimodular loci.
- Behavior near coordinate-degeneration strata.
- Behavior on fixed-point or pseudo-origin-type loci.
- Gradient and Hessian probes.
- Residual maps.
- Local asymptotic fits.
- Failure-mode analysis by stratum, patch, invariant coordinate, and curvature concentration.
- Direct empirical evaluation of the Mirjanic-Mishra Section 6 conjectures and propositions on the project's trained models.
- Phase-dependence/falsification tests for the Proposition 3.3 feature criterion on K3 and quintic samples.
- Derivative-bound probes for `phi`, including the scale of `partial_i phi / Z_i`, `partial_i partial_i phi`, and `partial_i partial_j phi / (Z_i Z_j)` where numerically stable.
- Pseudo-origin and equimodular-locus metric comparisons against the predicted formulae.
- Near-equimodular error-versus-distance curves for scaled-Fubini-Study approximations.
- Candidate constraints, sampling refinements, or model features suggested by the probes.
- A feedback report stating whether Milestone 5 should be rerun with modifications.

Completion criteria:

- The report identifies concrete candidate invariant variables, local forms, constraints, or sampling changes suggested by the numerical model.
- Each imported proposition/conjecture is classified as supported, violated, inconclusive, or not yet testable under the current numerical accuracy.
- Probing outputs are saved and reproducible.
- At least one decision is made explicitly:
  - iterate back to Milestone 5 with improved model/training design; or
  - proceed to Milestone 7 because the current numerical reference is accurate and well understood enough for quotient discovery.

## Milestone 5/6 Feedback Loop

Milestones 5 and 6 are intentionally iterative. If Milestone 6 identifies a reliable structural clue, difficult stratum, missing invariant, bad sampling region, or useful locus constraint, the next goal may return to Milestone 5 rather than proceeding linearly. Each loop should record:

- what Milestone 6 discovered;
- what changed in the Milestone 5 model, loss, sampling, or constraints;
- whether benchmark metrics improved;
- whether the new model is a better target for symbolic discovery.

## Milestone 7: Quotient Sufficiency And Invariant Basis Discovery

Determine whether the candidate quotient variables are sufficient to describe the numerical potential.

Deliverables:

- Quotient-collapse diagnostics using the trained numerical model.
- Comparison of invariant bases:

  ```text
  power sums p_k
  elementary symmetric polynomials e_k
  I_k-style invariants
  phase-sensitive invariants if required
  ```

- Diagnostics showing where each invariant basis succeeds or fails.
- A minimal sufficient invariant set, or explicit evidence that the current quotient is incomplete.

Completion criteria:

- The project can say, with numerical evidence, whether the potential is effectively a function of the chosen invariant coordinates.
- Missing invariants are documented if quotient collapse fails.

## Milestone 8: Structured Symbolic Ansatz Search

Fit interpretable candidate expansions against the high-accuracy numerical model and true geometric residuals.

Deliverables:

- Log-product models:

  ```text
  K = c0 + sum_a p_a log I_a
  ```

- Positive log-sum monomial models:

  ```text
  K = c0 + log(sum_m c_m prod_a I_a^{p_ma}), c_m > 0
  ```

- Sparse correction expansions:

  ```text
  K = K_structured + G(I)
  ```

- Rational exponent search.
- Sparse regression, PySR, CMA-ES, SINDy-style search, or related basis discovery.
- Re-optimization against true Monge-Ampere residuals after fitting to the numerical model.

Completion criteria:

- Candidate symbolic forms are saved in a candidate registry.
- Each candidate is evaluated on data fit, geometric residuals, positivity, symmetry, patch consistency, and simplicity.

## Milestone 9: Locus-Constrained Symbolic Discovery

Use known or discovered behavior on special loci as hard constraints.

Deliverables:

- Equimodular constraints.
- Fixed-locus constraints.
- Degeneration or asymptotic constraints.
- Ansatz form:

  ```text
  K = K_known + D(I)^2 psi(I)
  ```

  where `D(I)` vanishes on the constrained locus.

- Constrained versus unconstrained symbolic fit comparison.
- Theory-derived hard constraints from the Mirjanic-Mishra analytical properties when the current experiments support them:
  - exact pseudo-origin metric normalization;
  - equimodular-locus matching;
  - phase-independence restrictions;
  - derivative-growth envelopes;
  - vanishing factors for deviations away from constrained loci.

Completion criteria:

- Constraints are implemented as hard constraints where possible, not only penalties.
- The report states whether constraints improved numerical fit, geometric residuals, or interpretability.
- Any conjectural constraint is labelled as conjectural and backed by the Milestone 6 evidence before being used to restrict symbolic candidates.

## Milestone 10: Candidate Formula Evaluation

Treat each symbolic expression as a serious mathematical candidate.

Deliverables:

- Candidate formula registry.
- Data fit metrics.
- Monge-Ampere residual metrics.
- Positivity statistics.
- Symmetry consistency.
- Patch consistency.
- Failure-stratum analysis.
- Simplicity and complexity scores.
- Human-readable derivation notes.

Completion criteria:

- Good and bad candidates are both retained when their behavior is informative.
- The report clearly separates numerical evidence from conjectural mathematical claims.

## Milestone 11: Cross-Geometry Generalization

Test whether the K3 and Fermat quintic strategies transfer to related Calabi-Yau examples.

Deliverables:

- Additional Calabi-Yau geometries beyond Fermat quartic K3 and Fermat quintic.
- Numerical potential model or imported reference target for each new geometry.
- Invariant-basis comparison across geometries.
- Reuse of K3 architectures and ansatz families where appropriate.
- Notes on what appears universal versus K3-specific.

Completion criteria:

- At least one geometry beyond the quartic K3 and Fermat quintic runs through the numerical and symbolic pipeline.
- The report identifies which ideas survive generalization.

## Milestone 12: Analytical Interpretation

Translate empirical formulas into mathematical claims, conjectures, or proof obligations.

Deliverables:

- Candidate closed-form expression or convergent expansion.
- Evidence for stability under resampling, model changes, and basis enlargement.
- Relation to known algebraic, symmetric, or special-function structures.
- Explicit conjectures or proposition-style statements.
- List of open proof obligations.
- Draft paper section.

Completion criteria:

- The project has at least one serious candidate formula or expansion with a complete numerical evidence package.
- The report distinguishes proven facts, numerical evidence, conjectures, and open gaps.

## Mandatory Completion Checklist

Before marking any milestone complete, verify the following against the current repo state and current experiment outputs:

- [ ] Code implemented.
- [ ] Tests added or updated.
- [ ] Relevant experiment command run successfully.
- [ ] Numerical outputs saved.
- [ ] Figures and tables regenerated if affected.
- [ ] `reports/research_log.md` updated.
- [ ] `reports/paper.md` or `reports/paper.tex` updated if the result changes the research story.
- [ ] `MILESTONES.md` updated.
- [ ] Completed milestone checked off.
- [ ] New promising directions added as future milestones or submilestones.
- [ ] Known limitations explicitly documented.
- [ ] Best next milestone suggested.
- [ ] Any saved model, checkpoint, config, or artifact needed for reproducibility committed or placed in the agreed artifact path.
- [ ] Results verified from real current outputs rather than memory, intent, or stale reports.

## Suggested Next Goal

The next milestone should be Milestone 6:

```text
/goal Use /Users/ms/calabi-yau-lab as the base repo. Implement Milestone 6 from MILESTONES.md: numerical model probing and failure-mode analysis for the Milestone 5 K3 and quintic checkpoints. Probe equimodular, coordinate-degenerate, fixed-type, and pseudo-origin-type loci; evaluate residuals, gradients, Hessians, local asymptotics, invariant-coordinate maps, and Mirjanic-Mishra Proposition 3.3/Section 6 claims; classify each imported conjecture as supported, violated, inconclusive, or not yet testable; identify concrete sampling, constraint, or architecture changes; generate many figures and tables; update docs; and recommend whether to iterate back to Milestone 5 or proceed to quotient sufficiency discovery.
```
