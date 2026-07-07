from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cy_expansion_lab.config import ensure_dir, load_config
from cy_expansion_lab.cy.fermat import from_config
from cy_expansion_lab.cy.sampling import sample_fermat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fermat_quartic.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = ensure_dir(cfg["outputs"]["experiment_dir"])
    cy = from_config(cfg["target"])
    samples = sample_fermat(
        cy,
        n_samples=int(cfg["n_samples"]),
        seed=int(cfg["seed"]),
        train_fraction=float(cfg["splits"]["train"]),
        val_fraction=float(cfg["splits"]["val"]),
        include_orbits=bool(cfg["include_orbits"]),
    )
    np.savez_compressed(out / "samples.npz", z=samples.z, strata=samples.strata, split=samples.split)
    (out / "metadata.json").write_text(json.dumps(samples.metadata, indent=2), encoding="utf-8")
    print(f"Wrote {len(samples.z)} samples to {out}")


if __name__ == "__main__":
    main()
