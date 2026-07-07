from __future__ import annotations

import numpy as np

from cy_expansion_lab.fit.optimizers import mae, mse
from cy_expansion_lab.models.baseline import ConstantBaseline
from cy_expansion_lab.models.log_product import LogProductModel, PositiveLogSumMonomialModel


def fit_default_models(x: np.ndarray, y: np.ndarray, split: np.ndarray, n_terms: int) -> dict:
    train = split == "train"
    models = {
        "constant_baseline": ConstantBaseline(),
        "log_product": LogProductModel(),
        "positive_log_sum_monomial": PositiveLogSumMonomialModel(n_terms=n_terms),
    }
    results = {}
    for name, model in models.items():
        model.fit(x[train], y[train])
        results[name] = {"model": model, "metrics": {}}
        for part in ["train", "val", "test"]:
            mask = split == part
            pred = model.predict(x[mask])
            results[name]["metrics"][part] = {"mse": mse(y[mask], pred), "mae": mae(y[mask], pred)}
    return results
