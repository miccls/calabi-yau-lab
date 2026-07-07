from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cy_expansion_lab.config import ensure_dir, load_config
from cy_expansion_lab.cy.fermat import from_config
from cy_expansion_lab.fit.k3_potential import train_invariant_potential
from cy_expansion_lab.models.invariant_potential import DEFAULT_BASIS_NAMES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fermat_quartic.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if cfg["target"] != "fermat_quartic":
        raise ValueError("Milestone 2 training currently targets the Fermat quartic K3")
    out = ensure_dir(cfg["outputs"]["experiment_dir"])
    samples = np.load(out / "samples.npz", allow_pickle=True)
    split = samples["split"]
    z = samples["z"]
    train_z = z[split == "train"]
    val_z = z[split != "train"]
    cy = from_config(cfg["target"])
    training_cfg = cfg.get("training", {}).get("invariant_potential", {})
    basis_names = tuple(training_cfg.get("basis_names", list(DEFAULT_BASIS_NAMES)))
    model, metadata = train_invariant_potential(
        z_train=train_z,
        z_val=val_z,
        cy=cy,
        basis_names=basis_names,
        max_train_points=int(training_cfg.get("max_train_points", 36)),
        max_val_points=int(training_cfg.get("max_val_points", 48)),
        step=float(cfg.get("validation", {}).get("finite_difference_step", 1e-4)),
        seed=int(cfg["seed"]),
        maxiter=int(training_cfg.get("maxiter", 80)),
    )
    model_path = Path(training_cfg.get("model_path", "artifacts/models/fermat_quartic_invariant_potential.json"))
    metadata["config"] = args.config
    metadata["model_path"] = str(model_path)
    model.metadata = metadata
    model.save(model_path)
    (out / "trained_invariant_potential.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(_compact_metadata(metadata), indent=2))


def _compact_metadata(metadata: dict) -> dict:
    compact = dict(metadata)
    if "history" in compact:
        compact["history_length"] = len(compact.pop("history"))
    return compact


if __name__ == "__main__":
    main()
