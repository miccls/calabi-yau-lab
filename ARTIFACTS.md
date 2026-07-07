# Artifact Policy

This repo should stay reproducible without committing large generated outputs by default.

## Tracked By Default

- Source code under `src/`.
- Experiment scripts under `experiments/`.
- Small YAML configs under `configs/`.
- Tests under `tests/`.
- Research docs under `README.md`, `MILESTONES.md`, `CHECKLIST.md`, and `reports/`.
- Small, human-readable result summaries when they are part of the research record.
- Promoted benchmark tables under `reports/tables/`.
- Promoted milestone figures under `reports/figures/milestone*_*.png`.

## Ignored By Default

- Generated samples in `data/raw/`, `data/processed/`, and `data/experiments/`.
- Generated figures in `reports/figures/`.
- Local virtual environments and caches.
- Large model checkpoints.
- External datasets.

## Permanent Artifacts

When a model, checkpoint, dataset, or generated output becomes scientifically important, promote it deliberately:

1. Add a small metadata file describing how it was produced.
2. Record the command, config path, git commit, date, and machine notes.
3. Put small metadata in Git.
4. Put large binary artifacts under `artifacts/models/`, `artifacts/checkpoints/`, or `artifacts/external/`, or move them to external storage if they are too large for Git.
5. Link the artifact from `reports/research_log.md` and the relevant report section.

## Current First Experiment

The minimal quartic smoke experiment is reproducible with:

```bash
uv run python experiments/run_all.py --config configs/fermat_quartic.yaml
```

Generated outputs land in:

```text
data/experiments/fermat_quartic_smoke/
```

The current promoted small model artifact is:

```text
artifacts/models/fermat_quartic_invariant_potential.json
artifacts/models/fermat_quartic_milestone3_best.json
```

These are intentionally tracked because they are small, human-readable JSON files recording promoted geometry-trained K3 checkpoints.

## Current Milestone 4 Benchmark

The autodiff benchmark is reproducible with:

```bash
uv run --extra dev python experiments/run_milestone4.py --max-points 36 --sample-count 120 --ricci-points 1
```

Generated sample-level outputs land in:

```text
data/experiments/milestone4_benchmark/
```

Promoted small outputs are tracked in:

```text
reports/tables/milestone4_metrics.csv
reports/tables/milestone4_metric_definitions.csv
reports/tables/milestone4_theory_registry.csv
reports/figures/milestone4_*.png
reports/milestone4_section.md
reports/milestone4_log.md
```
