from __future__ import annotations

import numpy as np

from cy_expansion_lab.models.log_product import LogProductModel, PositiveLogSumMonomialModel


def test_log_product_recovers_log_linear_target() -> None:
    rng = np.random.default_rng(5)
    x = rng.uniform(0.1, 2.0, size=(80, 3))
    y = 0.3 + 1.2 * np.log(x[:, 0]) - 0.5 * np.log(x[:, 1]) + 0.2 * np.log(x[:, 2])
    model = LogProductModel().fit(x, y)
    assert np.mean((model.predict(x) - y) ** 2) < 1e-20


def test_positive_log_sum_runs() -> None:
    rng = np.random.default_rng(6)
    x = rng.uniform(0.1, 2.0, size=(60, 4))
    y = np.log(1.0 + 0.7 * x[:, 0] + 0.2 * x[:, 1])
    model = PositiveLogSumMonomialModel(n_terms=3).fit(x, y)
    assert np.isfinite(model.predict(x)).all()
