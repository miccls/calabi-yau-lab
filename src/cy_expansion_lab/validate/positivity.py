from __future__ import annotations

import numpy as np


def positive_expression_proxy(pred: np.ndarray) -> dict[str, float]:
    values = np.exp(pred)
    return {
        "min_exp_potential": float(np.min(values)),
        "nonfinite_rate": float(np.mean(~np.isfinite(values))),
    }
