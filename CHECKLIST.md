# Milestone Completion Checklist

Use this checklist before marking any milestone in `MILESTONES.md` complete.

## Required Checks

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
- [ ] Any saved model, checkpoint, config, or artifact needed for reproducibility is committed or placed in the agreed artifact path.
- [ ] Results verified from real current outputs rather than memory, intent, or stale reports.

## Milestone 0 Specific Checks

- [x] Canonical `calabi-yau-lab` repository exists.
- [x] Existing `cy_expansion_lab` work is migrated or incorporated.
- [x] Git repository initialized locally.
- [x] GitHub-ready `.gitignore` exists.
- [x] Artifact policy exists.
- [x] README documents setup and first experiment command.
- [x] Minimal quartic experiment runs from the canonical repo path.
- [x] Tests pass from the canonical repo path.

## Suggested Next Milestone

After Milestone 5, the next milestone should be Milestone 6: numerical model probing and failure-mode analysis.

## Milestone 1 Specific Checks

- [x] Affine Fermat quartic K3 patches implemented.
- [x] Hypersurface constraint reconstruction implemented and tested.
- [x] Pullback of projective/Fubini-Study potential to local coordinates implemented.
- [x] Complex Hessian computation implemented and tested.
- [x] Metric eigenvalue positivity checks implemented.
- [x] Holomorphic volume density implemented.
- [x] True centered Monge-Ampere residual diagnostics implemented.
- [x] Validation script reports true local geometry diagnostics.
- [x] Tests pass from the canonical repo path.
- [x] Minimal quartic experiment runs from the canonical repo path.
- [x] Research log and report updated with implementation details and limitations.

## Milestone 2 Specific Checks

- [x] Symmetry-aware K3 potential model implemented.
- [x] Kähler-potential formulation preserved.
- [x] Projective, Fermat phase, and permutation symmetries enforced by invariant features.
- [x] Gauge handled by using a correction to Fubini-Study plus optional constant shift.
- [x] Training objective uses true local Monge-Ampere residuals and a positivity penalty.
- [x] Validation reports held-out local geometry metrics.
- [x] Saved checkpoint exists at `artifacts/models/fermat_quartic_invariant_potential.json`.
- [x] Model beats Fubini-Study/reference baseline on geometric residuals in the verified run.
- [x] Tests pass from the canonical repo path.
- [x] Minimal quartic experiment runs from the canonical repo path.
- [x] README, research log, report, milestones, and checklist updated.

## Milestone 3 Specific Checks

- [x] Stronger invariant basis implemented.
- [x] Multi-seed and hyperparameter/config sweep implemented.
- [x] Best model selected by held-out geometric metrics.
- [x] Model improves over Milestone 2 checkpoint on held-out centered log-Monge-Ampere RMSE.
- [x] Best checkpoint saved at `artifacts/models/fermat_quartic_milestone3_best.json`.
- [x] Training curves generated.
- [x] Validation Monge-Ampere/residual figures generated.
- [x] Metric positivity/min-eigenvalue figure generated.
- [x] Residual histograms generated.
- [x] Stratum-wise residual plot generated.
- [x] Correction-versus-invariant plot generated.
- [x] Comparison bar chart generated.
- [x] Metric definitions and units/conventions documented.
- [x] Literature comparison table included without mixing incompatible units.
- [x] Tests pass from the canonical repo path.
- [x] Milestone 3 command runs from the canonical repo path.
- [x] Standard quartic smoke command still runs.
- [x] README, research log, report, milestones, checklist, and artifact policy updated.

## Milestone 4 Specific Checks

- [x] Autodiff backend added and documented.
- [x] Autodiff Kähler-potential derivative path implemented.
- [x] Autodiff Hermitian metric computation implemented.
- [x] Autodiff determinant/log-determinant computation implemented.
- [x] Autodiff Monge-Ampere residual computation implemented.
- [x] Metric positivity and minimum-eigenvalue diagnostics work with the autodiff path.
- [x] Ricci tensor and/or scalar-curvature diagnostics implemented, or a documented computational reason is given for deferring them.
- [x] Autodiff Hessians/log determinants tested against finite-difference K3 diagnostics on small samples.
- [x] Published metric/loss module implemented.
- [x] Centered log-Monge-Ampere RMSE/MAE definitions documented.
- [x] Raw Monge-Ampere volume-ratio error definition documented.
- [x] Sigma measure / sigma loss convention documented and implemented as closely as possible to the literature.
- [x] Ricci diagnostic units and aggregation conventions documented.
- [x] Kähler and transition/patch consistency diagnostics implemented where relevant.
- [x] Training objectives are separated from evaluation metrics in code and reports.
- [x] Metric-definition table generated in machine-readable form.
- [x] Metric-definition table included in the report.
- [x] Literature-theory registry implemented for propositions, conjectures, and paper-derived constraints.
- [x] Mirjanic-Mishra Proposition 3.3 recorded with source, formula, implementation status, and assumption/proof status.
- [x] Mirjanic-Mishra Section 6 analytical properties of `phi` recorded with source, formula, implementation status, and assumption/proof status.
- [x] Theory registry distinguishes proven results, conjectures, empirical observations, and assumptions.
- [x] Fermat quintic point sampling implemented or imported through a documented dependency.
- [x] Fermat quintic local patch reconstruction implemented.
- [x] Fermat quintic holomorphic volume density implemented.
- [x] Fermat quintic Fubini-Study baseline evaluation runs successfully.
- [x] Benchmark runner evaluates Fubini-Study and current K3 checkpoints.
- [x] Benchmark figures generated with units and aggregation conventions in labels/captions.
- [x] Tests pass from the canonical repo path.
- [x] Benchmark command runs from the canonical repo path.
- [x] README, research log, report, milestones, checklist, and artifact policy updated.

## Milestone 5 Specific Checks

- [x] Autodiff-trained K3 model implemented.
- [x] Fermat quintic model training implemented.
- [x] Geometry-aware and symmetry-aware architecture choices documented.
- [x] Loss ablations implemented for the relevant published losses.
- [x] Stratified or difficult-locus sampling implemented.
- [x] Locus constraints implemented if reliable formulas are available, or explicitly deferred.
- [x] Phase-invariance features or restrictions from Mirjanic-Mishra Proposition 3.3 evaluated for training use.
- [x] Derivative-bound regularizers inspired by Section 6 are explicitly deferred to Milestone 6 pending stable derivative probes.
- [x] Pseudo-origin metric target losses from Section 6 are explicitly deferred to Milestone 6 pending a compatible locus implementation.
- [x] Equimodular-locus metric target losses from Section 6 are explicitly deferred to Milestone 6 pending formula/convention validation.
- [x] Near-equimodular scaled-Fubini-Study warm start or auxiliary loss is explicitly deferred to Milestone 6 probing.
- [x] Simple `sqrt(s2)` correction baseline from Section 6 is evaluated against project metrics.
- [x] Theory-derived training constraints are labelled as hard constraints, soft penalties, feature restrictions, warm starts, or evaluation-only diagnostics.
- [x] Multi-seed sweep implemented.
- [x] Hyperparameter and architecture sweep implemented.
- [x] Best model selected by held-out benchmark metrics.
- [x] K3 best model improves over the Milestone 3 checkpoint, or the failure is analyzed.
- [x] Quintic result is reported with the same metric definitions as K3.
- [x] Training curves generated.
- [x] Benchmark comparison figures generated.
- [x] Residual histograms and stratum/locus residual figures generated.
- [x] Literature comparison table uses compatible units only.
- [x] Best checkpoints, configs, seeds, and exact commands saved permanently.
- [x] Tests pass from the canonical repo path.
- [x] SOTA training command runs from the canonical repo path.
- [x] README, research log, report, milestones, checklist, and artifact policy updated.

## Milestone 6 Specific Checks

- [ ] Best Milestone 5 K3 model loaded and probed.
- [ ] Best Milestone 5 quintic model loaded and probed if available.
- [ ] Equimodular locus probes implemented.
- [ ] Coordinate-degeneration stratum probes implemented.
- [ ] Fixed-point or pseudo-origin-type locus probes implemented where applicable.
- [ ] Gradient and Hessian probes implemented.
- [ ] Residual maps generated.
- [ ] Local asymptotic fits generated.
- [ ] Failure modes analyzed by stratum, patch, invariant coordinate, or curvature concentration.
- [ ] Proposition 3.3 phase-dependence/falsification tests run on K3 and quintic samples.
- [ ] Section 6 cross-dimensional self-similarity conjecture tested where comparable K3/quintic data exists.
- [ ] Section 6 derivative-bound probes run where numerically stable.
- [ ] Pseudo-origin metric formula tested against trained model outputs.
- [ ] Equimodular-locus metric formula tested against trained model outputs.
- [ ] Near-equimodular scaled-Fubini-Study approximation error plotted against distance from the locus.
- [ ] Each imported proposition/conjecture classified as supported, violated, inconclusive, or not yet testable.
- [ ] Candidate constraints, sampling refinements, or model features are extracted from the probes.
- [ ] A decision is recorded: iterate back to Milestone 5 or proceed to Milestone 7.
- [ ] If iterating back to Milestone 5, the proposed model/loss/sampling/constraint change is written down.
- [ ] Probing outputs and figures saved reproducibly.
- [ ] Tests pass from the canonical repo path.
- [ ] Probing command runs from the canonical repo path.
- [ ] README, research log, report, milestones, checklist, and artifact policy updated.
