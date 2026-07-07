from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass
class LogProductModel:
    eps: float = 1e-10
    coef_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LogProductModel":
        logx = np.log(np.maximum(x, self.eps))
        design = np.concatenate([np.ones((x.shape[0], 1)), logx], axis=1)
        self.coef_, *_ = np.linalg.lstsq(design, y, rcond=None)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted")
        logx = np.log(np.maximum(x, self.eps))
        design = np.concatenate([np.ones((x.shape[0], 1)), logx], axis=1)
        return design @ self.coef_


@dataclass
class PositiveLogSumMonomialModel:
    n_terms: int = 4
    eps: float = 1e-10
    params_: np.ndarray | None = None
    exponents_: np.ndarray | None = None

    def _basis(self, x: np.ndarray) -> np.ndarray:
        if self.exponents_ is None:
            raise RuntimeError("Exponents are not initialized")
        logx = np.log(np.maximum(x, self.eps))
        return np.exp(logx @ self.exponents_.T)

    def _predict_from_params(self, params: np.ndarray, x: np.ndarray) -> np.ndarray:
        c0 = params[0]
        weights = np.exp(params[1:])
        return c0 + np.log(self._basis(x) @ weights + self.eps)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PositiveLogSumMonomialModel":
        n_features = x.shape[1]
        rng = np.random.default_rng(13)
        eye_terms = np.eye(n_features)[: min(self.n_terms, n_features)]
        if len(eye_terms) < self.n_terms:
            extra = rng.uniform(-1.0, 1.0, size=(self.n_terms - len(eye_terms), n_features))
            self.exponents_ = np.vstack([eye_terms, extra])
        else:
            self.exponents_ = eye_terms
        init = np.zeros(1 + self.n_terms)

        def residual(params: np.ndarray) -> np.ndarray:
            return self._predict_from_params(params, x) - y

        res = least_squares(residual, init, max_nfev=2000)
        self.params_ = res.x
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.params_ is None:
            raise RuntimeError("Model is not fitted")
        return self._predict_from_params(self.params_, x)
