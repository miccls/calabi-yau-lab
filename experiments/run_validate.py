from __future__ import annotations

import argparse
import json

import numpy as np

from cy_expansion_lab.config import ensure_dir, load_config
from cy_expansion_lab.cy.fermat import from_config
from cy_expansion_lab.fit.train import fit_default_models
from cy_expansion_lab.models.invariant_potential import InvariantPotentialModel
from cy_expansion_lab.validate.local_geometry import (
    fs_geometry_diagnostics,
    invariant_potential_geometry_diagnostics,
    model_geometry_diagnostics,
)
from cy_expansion_lab.validate.monge_ampere import volume_ratio_proxy
from cy_expansion_lab.validate.positivity import positive_expression_proxy
from cy_expansion_lab.validate.symmetry import orbit_consistency


def _compact_metadata(metadata: dict) -> dict:
    compact = dict(metadata)
    if "history" in compact:
        compact["history_length"] = len(compact.pop("history"))
    return compact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fermat_quartic.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = ensure_dir(cfg["outputs"]["experiment_dir"])
    samples = np.load(out / "samples.npz", allow_pickle=True)
    inv = np.load(out / "invariants.npz")
    cy = from_config(cfg["target"])
    x = inv["positive_features"]
    feature_min = np.min(inv["features"], axis=0)
    y = inv["target"]
    split = samples["split"]
    geometry_cfg = cfg.get("validation", {})
    geometry_max_points = int(geometry_cfg.get("geometry_max_points", 48))
    fd_step = float(geometry_cfg.get("finite_difference_step", 1e-4))
    results = fit_default_models(x, y, split, n_terms=int(cfg["models"]["log_sum_terms"]))
    validation = {
        "fubini_study_reference": {
            "local_geometry": fs_geometry_diagnostics(
                samples["z"], cy=cy, max_points=geometry_max_points, step=fd_step
            )
        }
    }
    trained_model_path = cfg.get("training", {}).get("invariant_potential", {}).get("model_path")
    if trained_model_path:
        try:
            trained_model = InvariantPotentialModel.load(trained_model_path)
            validation["trained_invariant_potential"] = {
                "checkpoint": trained_model_path,
                "metadata": _compact_metadata(trained_model.metadata or {}),
                "local_geometry": invariant_potential_geometry_diagnostics(
                    samples["z"], cy=cy, model=trained_model, max_points=geometry_max_points, step=fd_step
                ),
            }
        except FileNotFoundError:
            validation["trained_invariant_potential"] = {
                "checkpoint": trained_model_path,
                "missing": True,
            }
    for name, payload in results.items():
        pred_all = payload["model"].predict(x)
        validation[name] = {
            "positivity_proxy": positive_expression_proxy(pred_all),
            "symmetry_consistency": orbit_consistency(pred_all, group_size=3),
            "volume_ratio_proxy": volume_ratio_proxy(pred_all, inv["log_omega_proxy"]),
            "local_geometry": model_geometry_diagnostics(
                samples["z"],
                cy=cy,
                model=payload["model"],
                feature_min=feature_min,
                power_max=int(cfg["invariants"]["power_max"]),
                max_points=geometry_max_points,
                step=fd_step,
            ),
        }
    (out / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
