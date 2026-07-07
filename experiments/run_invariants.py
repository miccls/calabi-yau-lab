from __future__ import annotations

import argparse
import json

import numpy as np

from cy_expansion_lab.config import ensure_dir, load_config
from cy_expansion_lab.cy.holomorphic_volume import fermat_holomorphic_volume_proxy
from cy_expansion_lab.cy.projective import normalized_radii
from cy_expansion_lab.invariants.quotient import invariant_table


def synthetic_invariant_target(features: np.ndarray) -> np.ndarray:
    """Stable proxy for a structured correction potential."""
    x = features - np.min(features, axis=0, keepdims=True) + 1e-6
    return np.log(0.75 + 1.6 * x[:, 0] + 0.8 * np.sqrt(x[:, 1]) + 0.35 * x[:, 2] * x[:, 3])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fermat_quartic.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = ensure_dir(cfg["outputs"]["experiment_dir"])
    samples = np.load(out / "samples.npz", allow_pickle=True)
    z = samples["z"]
    features, names = invariant_table(z, power_max=int(cfg["invariants"]["power_max"]))
    r = normalized_radii(z)
    target = synthetic_invariant_target(features)
    log_omega_proxy = fermat_holomorphic_volume_proxy(z, degree=int(json.loads((out / "metadata.json").read_text())["degree"]))
    np.savez_compressed(
        out / "invariants.npz",
        features=features,
        positive_features=features - np.min(features, axis=0, keepdims=True) + 1e-6,
        r=r,
        target=target,
        log_omega_proxy=log_omega_proxy,
    )
    (out / "invariant_names.json").write_text(json.dumps(names, indent=2), encoding="utf-8")
    print(f"Wrote {features.shape[1]} invariant features to {out}")


if __name__ == "__main__":
    main()
