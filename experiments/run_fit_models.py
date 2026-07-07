from __future__ import annotations

import argparse
import json

import numpy as np

from cy_expansion_lab.config import ensure_dir, load_config
from cy_expansion_lab.fit.train import fit_default_models
from cy_expansion_lab.reporting.plots import predicted_vs_target


def _jsonable_metrics(results: dict) -> dict:
    return {name: payload["metrics"] for name, payload in results.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fermat_quartic.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = ensure_dir(cfg["outputs"]["experiment_dir"])
    samples = np.load(out / "samples.npz", allow_pickle=True)
    inv = np.load(out / "invariants.npz")
    x = inv["positive_features"]
    y = inv["target"]
    split = samples["split"]
    results = fit_default_models(x, y, split, n_terms=int(cfg["models"]["log_sum_terms"]))
    metrics = _jsonable_metrics(results)
    (out / "fit_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    mask = split == "test"
    for name, payload in results.items():
        pred = payload["model"].predict(x[mask])
        predicted_vs_target(out / f"{name}_test_scatter.png", y[mask], pred, name)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
