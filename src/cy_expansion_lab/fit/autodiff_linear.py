from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from cy_expansion_lab.cy.autodiff import (
    complex_hessian_autodiff,
    fs_patch_potential_jax,
    invariant_basis_jax,
    reconstruct_affine_jax,
)
from cy_expansion_lab.cy.fermat import FermatHypersurface
from cy_expansion_lab.cy.local_patch import FermatLocalPatch
from cy_expansion_lab.models.invariant_potential import InvariantPotentialModel
from cy_expansion_lab.validate.benchmark_metrics import summarize_log_ma


SelectionMode = Literal["stratified", "linspace"]


@dataclass(frozen=True)
class LossWeights:
    centered_log_ma: float = 1.0
    volume_ratio_mse: float = 0.0
    sigma_l1_smooth: float = 0.0
    positivity: float = 20.0
    l2: float = 1e-4
    eig_floor: float = 1e-7

    def as_dict(self) -> dict[str, float]:
        return {
            "centered_log_ma": self.centered_log_ma,
            "volume_ratio_mse": self.volume_ratio_mse,
            "sigma_l1_smooth": self.sigma_l1_smooth,
            "positivity": self.positivity,
            "l2": self.l2,
            "eig_floor": self.eig_floor,
        }


@dataclass(frozen=True)
class AutodiffLinearTrainingSet:
    target: str
    basis_names: tuple[str, ...]
    indices: np.ndarray
    strata: np.ndarray
    fs_metrics: np.ndarray
    basis_hessians: np.ndarray
    omega_density: np.ndarray
    patch_residuals: np.ndarray

    @property
    def basis_size(self) -> int:
        return len(self.basis_names)

    @property
    def count(self) -> int:
        return int(len(self.indices))


def build_autodiff_linear_training_set(
    z: np.ndarray,
    strata: np.ndarray,
    cy: FermatHypersurface,
    basis_names: tuple[str, ...],
    max_points: int,
    selection: SelectionMode = "stratified",
) -> AutodiffLinearTrainingSet:
    indices = select_patch_indices(strata, max_points=max_points, mode=selection)
    fs_metrics = []
    basis_hessians = []
    omega = []
    residuals = []
    used_indices = []
    used_strata = []
    for idx in indices:
        try:
            patch, u0 = FermatLocalPatch.from_point(cy, z[idx])
            fs_metric = complex_hessian_autodiff(fs_patch_potential_jax(patch), u0)
            term_hessians = [
                complex_hessian_autodiff(_basis_term_potential(patch, basis_names, term_idx), u0)
                for term_idx in range(len(basis_names))
            ]
            fs_metrics.append(fs_metric)
            basis_hessians.append(np.stack(term_hessians, axis=0))
            omega.append(patch.holomorphic_volume_density(u0))
            residuals.append(abs(patch.defining_residual(u0)))
            used_indices.append(int(idx))
            used_strata.append(str(strata[idx]))
        except (FloatingPointError, ValueError, OverflowError):
            continue
    if not fs_metrics:
        raise ValueError("No valid local patches were available for autodiff linear training")
    return AutodiffLinearTrainingSet(
        target=cy.name,
        basis_names=basis_names,
        indices=np.array(used_indices, dtype=int),
        strata=np.array(used_strata, dtype=object),
        fs_metrics=np.stack(fs_metrics, axis=0),
        basis_hessians=np.stack(basis_hessians, axis=0),
        omega_density=np.array(omega, dtype=float),
        patch_residuals=np.array(residuals, dtype=float),
    )


def train_autodiff_linear_potential(
    train_set: AutodiffLinearTrainingSet,
    val_set: AutodiffLinearTrainingSet,
    cy: FermatHypersurface,
    loss_weights: LossWeights,
    seed: int,
    maxiter: int = 200,
    initial_model: InvariantPotentialModel | None = None,
) -> tuple[InvariantPotentialModel, dict]:
    rng = np.random.default_rng(seed)
    initial = _initial_coefficients(train_set.basis_names, initial_model)
    if initial_model is None:
        initial = 0.01 * rng.normal(size=train_set.basis_size)
    value_and_grad = _make_value_and_grad(train_set, loss_weights)
    history: list[dict] = []

    def objective(coefficients: np.ndarray) -> tuple[float, np.ndarray]:
        value, grad = value_and_grad(jnp.asarray(coefficients, dtype=jnp.float64))
        return float(value), np.asarray(grad, dtype=float)

    def record(stage: str, coefficients: np.ndarray) -> None:
        train_metrics = evaluate_autodiff_coefficients(train_set, coefficients, loss_weights=loss_weights)
        val_metrics = evaluate_autodiff_coefficients(val_set, coefficients, loss_weights=loss_weights)
        history.append(
            {
                "step": len(history),
                "stage": stage,
                "objective": train_metrics["loss"]["total"],
                "train_centered_log_ma_rmse": train_metrics["summary"]["centered_log_ma_rmse"],
                "validation_centered_log_ma_rmse": val_metrics["summary"]["centered_log_ma_rmse"],
                "train_sigma_l1_volume_ratio": train_metrics["summary"]["sigma_l1_volume_ratio"],
                "validation_sigma_l1_volume_ratio": val_metrics["summary"]["sigma_l1_volume_ratio"],
                "train_pos_fail": train_metrics["summary"]["positivity_violation_rate"],
                "validation_pos_fail": val_metrics["summary"]["positivity_violation_rate"],
                "validation_min_eigenvalue": val_metrics["summary"]["min_eigenvalue_min"],
            }
        )

    record("initial", initial)

    def scipy_fun(coefficients: np.ndarray) -> float:
        value, _ = objective(coefficients)
        return value

    def scipy_jac(coefficients: np.ndarray) -> np.ndarray:
        _, grad = objective(coefficients)
        return grad

    result = minimize(
        scipy_fun,
        initial,
        jac=scipy_jac,
        method="L-BFGS-B",
        bounds=[(-1.5, 1.5)] * train_set.basis_size,
        callback=lambda xk: record("lbfgs", np.asarray(xk, dtype=float)),
        options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-8, "maxls": 40},
    )
    coefficients = np.asarray(result.x, dtype=float)
    record("selected", coefficients)
    train_metrics = evaluate_autodiff_coefficients(train_set, coefficients, loss_weights=loss_weights)
    val_metrics = evaluate_autodiff_coefficients(val_set, coefficients, loss_weights=loss_weights)
    metadata = {
        "target": cy.name,
        "model_family": "autodiff_linear_invariant_correction",
        "basis_names": list(train_set.basis_names),
        "seed": int(seed),
        "optimizer": "scipy_l_bfgs_b_with_jax_value_and_grad",
        "maxiter": int(maxiter),
        "loss_weights": loss_weights.as_dict(),
        "train_patch_count": train_set.count,
        "validation_patch_count": val_set.count,
        "train_strata_counts": _strata_counts(train_set.strata),
        "validation_strata_counts": _strata_counts(val_set.strata),
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_iterations": int(result.nit),
        "history": history,
        "autodiff_backend": "jax",
    }
    return InvariantPotentialModel(coefficients=coefficients, basis_names=train_set.basis_names, metadata=metadata), metadata


def evaluate_autodiff_coefficients(
    training_set: AutodiffLinearTrainingSet,
    coefficients: np.ndarray,
    loss_weights: LossWeights | None = None,
) -> dict:
    metrics = metrics_from_coefficients(training_set, coefficients)
    eigvals = np.linalg.eigvalsh(metrics)
    min_eig = np.min(eigvals, axis=1)
    safe_eig = np.maximum(eigvals, 1e-15)
    logdet = np.sum(np.log(safe_eig), axis=1)
    raw_log_ma = np.where(np.all(eigvals > 0.0, axis=1), logdet - np.log(training_set.omega_density), np.nan)
    summary = summarize_log_ma(raw_log_ma, min_eig)
    records = pointwise_records_from_training_set(training_set, coefficients)
    payload: dict[str, object] = {"summary": summary, "records": records}
    if loss_weights is not None:
        payload["loss"] = _loss_components_numpy(training_set, coefficients, loss_weights)
    return payload


def metrics_from_coefficients(training_set: AutodiffLinearTrainingSet, coefficients: np.ndarray) -> np.ndarray:
    return training_set.fs_metrics + np.einsum("a,najk->njk", coefficients, training_set.basis_hessians)


def pointwise_records_from_training_set(training_set: AutodiffLinearTrainingSet, coefficients: np.ndarray) -> list[dict]:
    metrics = metrics_from_coefficients(training_set, coefficients)
    eigvals = np.linalg.eigvalsh(metrics)
    min_eigs = np.min(eigvals, axis=1)
    positive = np.all(eigvals > 0.0, axis=1)
    raw_values = []
    for i, is_positive in enumerate(positive):
        if is_positive:
            raw_values.append(float(np.sum(np.log(eigvals[i])) - np.log(training_set.omega_density[i])))
    center = float(np.mean(raw_values)) if raw_values else float("nan")
    records = []
    for pos, idx in enumerate(training_set.indices):
        raw_log_ma = None
        volume_ratio = None
        centered = None
        logdet = None
        if positive[pos]:
            logdet = float(np.sum(np.log(eigvals[pos])))
            raw_log_ma = float(logdet - np.log(training_set.omega_density[pos]))
            volume_ratio = float(np.exp(np.clip(raw_log_ma, -700.0, 700.0)))
            centered = float(raw_log_ma - center)
        records.append(
            {
                "index": int(idx),
                "target": training_set.target,
                "stratum": str(training_set.strata[pos]),
                "min_eigenvalue": float(min_eigs[pos]),
                "positive": bool(positive[pos]),
                "logdet": logdet,
                "omega_density": float(training_set.omega_density[pos]),
                "raw_log_ma": raw_log_ma,
                "volume_ratio": volume_ratio,
                "centered_log_ma_residual": centered,
                "patch_residual": float(training_set.patch_residuals[pos]),
            }
        )
    return records


def select_patch_indices(strata: np.ndarray, max_points: int, mode: SelectionMode = "stratified") -> np.ndarray:
    n = len(strata)
    if n <= max_points:
        return np.arange(n, dtype=int)
    if mode == "linspace":
        return np.linspace(0, n - 1, max_points, dtype=int)
    labels = np.array([str(s).split(":")[0] for s in strata], dtype=object)
    unique = sorted(set(labels))
    per = max(1, max_points // max(1, len(unique)))
    selected: list[int] = []
    for label in unique:
        idx = np.flatnonzero(labels == label)
        take = min(len(idx), per)
        if take:
            positions = np.linspace(0, len(idx) - 1, take, dtype=int)
            selected.extend(idx[positions].tolist())
    if len(selected) < max_points:
        remaining = [i for i in np.linspace(0, n - 1, max_points * 2, dtype=int).tolist() if i not in set(selected)]
        selected.extend(remaining[: max_points - len(selected)])
    return np.array(sorted(selected[:max_points]), dtype=int)


def _basis_term_potential(patch: FermatLocalPatch, basis_names: tuple[str, ...], term_idx: int):
    def potential(u: jnp.ndarray) -> jnp.ndarray:
        w = reconstruct_affine_jax(patch, u)
        return invariant_basis_jax(w, basis_names)[0, term_idx]

    return potential


def _make_value_and_grad(training_set: AutodiffLinearTrainingSet, weights: LossWeights):
    fs_metrics = jnp.asarray(training_set.fs_metrics, dtype=jnp.complex128)
    basis_hessians = jnp.asarray(training_set.basis_hessians, dtype=jnp.complex128)
    log_omega = jnp.log(jnp.asarray(training_set.omega_density, dtype=jnp.float64))
    eig_floor = float(weights.eig_floor)

    def objective(coefficients: jnp.ndarray) -> jnp.ndarray:
        metrics = fs_metrics + jnp.einsum("a,najk->njk", coefficients, basis_hessians)
        eigvals = jnp.linalg.eigvalsh(metrics)
        min_eigs = eigvals[:, 0]
        safe = jnp.maximum(eigvals, eig_floor)
        raw = jnp.sum(jnp.log(safe), axis=1) - log_omega
        centered = raw - jnp.mean(raw)
        ma_loss = jnp.mean(centered**2)
        shifted = raw - jnp.max(raw)
        ratio = jnp.exp(shifted) / jnp.mean(jnp.exp(shifted))
        volume_loss = jnp.mean((ratio - 1.0) ** 2)
        sigma_smooth = jnp.mean(jnp.sqrt((ratio - 1.0) ** 2 + 1e-8))
        positivity = jnp.mean(jnp.maximum(eig_floor - min_eigs, 0.0) ** 2)
        l2 = jnp.sum(coefficients**2)
        return (
            weights.centered_log_ma * ma_loss
            + weights.volume_ratio_mse * volume_loss
            + weights.sigma_l1_smooth * sigma_smooth
            + weights.positivity * positivity
            + weights.l2 * l2
        )

    return jax.value_and_grad(objective)


def _loss_components_numpy(training_set: AutodiffLinearTrainingSet, coefficients: np.ndarray, weights: LossWeights) -> dict[str, float]:
    metrics = metrics_from_coefficients(training_set, coefficients)
    eigvals = np.linalg.eigvalsh(metrics)
    min_eigs = eigvals[:, 0]
    safe = np.maximum(eigvals, weights.eig_floor)
    raw = np.sum(np.log(safe), axis=1) - np.log(training_set.omega_density)
    centered = raw - np.mean(raw)
    shifted = raw - np.max(raw)
    ratio = np.exp(shifted) / np.mean(np.exp(shifted))
    ma_loss = float(np.mean(centered**2))
    volume_loss = float(np.mean((ratio - 1.0) ** 2))
    sigma_smooth = float(np.mean(np.sqrt((ratio - 1.0) ** 2 + 1e-8)))
    positivity = float(np.mean(np.maximum(weights.eig_floor - min_eigs, 0.0) ** 2))
    l2 = float(np.sum(coefficients**2))
    return {
        "centered_log_ma_mse": ma_loss,
        "volume_ratio_mse": volume_loss,
        "sigma_l1_smooth": sigma_smooth,
        "positivity_penalty": positivity,
        "l2": l2,
        "total": (
            weights.centered_log_ma * ma_loss
            + weights.volume_ratio_mse * volume_loss
            + weights.sigma_l1_smooth * sigma_smooth
            + weights.positivity * positivity
            + weights.l2 * l2
        ),
    }


def _initial_coefficients(basis_names: tuple[str, ...], initial_model: InvariantPotentialModel | None) -> np.ndarray:
    coefficients = np.zeros(len(basis_names), dtype=float)
    if initial_model is None:
        return coefficients
    initial_lookup = dict(zip(initial_model.basis_names, initial_model.coefficients))
    for i, name in enumerate(basis_names):
        if name in initial_lookup:
            coefficients[i] = float(initial_lookup[name])
    return coefficients


def _strata_counts(strata: np.ndarray) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in strata:
        key = str(value)
        out[key] = out.get(key, 0) + 1
    return out
