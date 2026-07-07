from __future__ import annotations

import numpy as np


class ConstantBaseline:
    def __init__(self) -> None:
        self.value_: float = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "ConstantBaseline":
        del x
        self.value_ = float(np.mean(y))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full(x.shape[0], self.value_, dtype=float)
