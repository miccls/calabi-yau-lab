from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def predicted_vs_target(path: str | Path, y: np.ndarray, pred: np.ndarray, title: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    ax.scatter(y, pred, s=8, alpha=0.55)
    lo = min(float(np.min(y)), float(np.min(pred)))
    hi = max(float(np.max(y)), float(np.max(pred)))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
    ax.set_xlabel("target")
    ax.set_ylabel("prediction")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
