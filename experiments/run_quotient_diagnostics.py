from __future__ import annotations

import argparse
import json

import numpy as np

from cy_expansion_lab.config import ensure_dir, load_config
from cy_expansion_lab.invariants.quotient import quotient_collapse_error
from cy_expansion_lab.validate.symmetry import orbit_consistency


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fermat_quartic.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = ensure_dir(cfg["outputs"]["experiment_dir"])
    inv = np.load(out / "invariants.npz")
    y = inv["target"]
    diag = {
        "quotient_collapse": quotient_collapse_error(inv["features"], y, k=8),
        "target_orbit_consistency": orbit_consistency(y, group_size=3),
    }
    (out / "quotient_diagnostics.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")
    print(json.dumps(diag, indent=2))


if __name__ == "__main__":
    main()
