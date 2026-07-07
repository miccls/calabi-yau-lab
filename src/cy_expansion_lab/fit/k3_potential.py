from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import differential_evolution, minimize

from cy_expansion_lab.cy.fermat import FermatHypersurface
from cy_expansion_lab.cy.local_patch import (
    FermatLocalPatch,
    complex_hessian_from_potential,
    fs_affine_potential,
)
from cy_expansion_lab.models.invariant_potential import DEFAULT_BASIS_NAMES, InvariantPotentialModel, invariant_basis
from cy_expansion_lab.validate.kahler import hermitian_eigenvalue_stats
from cy_expansion_lab.validate.monge_ampere import monge_ampere_residual


@dataclass(frozen=True)
class LocalTrainingSet:
    fs_metrics: np.ndarray
    basis_hessians: np.ndarray
    omega_density: np.ndarray
    patch_residuals: np.ndarray


def build_local_training_set(
    z: np.ndarray,
    cy: FermatHypersurface,
    basis_names: tuple[str, ...] = DEFAULT_BASIS_NAMES,
    max_points: int = 32,
    step: float = 1e-4,
) -> LocalTrainingSet:
    indices = np.linspace(0, len(z) - 1, min(max_points, len(z)), dtype=int)
    fs_metrics = []
    basis_hessians = []
    omega = []
    residuals = []
    for idx in indices:
        patch, u0 = FermatLocalPatch.from_point(cy, z[idx])
        fs_metric = complex_hessian_from_potential(
            lambda u: fs_affine_potential(patch.reconstruct_affine(u)), u0, step=step
        )
        term_hessians = []
        for term_idx in range(len(basis_names)):
            term_hessians.append(
                complex_hessian_from_potential(
                    lambda u, j=term_idx: float(invariant_basis(patch.reconstruct_affine(u)[None, :], basis_names)[0, j]),
                    u0,
                    step=step,
                )
            )
        fs_metrics.append(fs_metric)
        basis_hessians.append(np.stack(term_hessians, axis=0))
        omega.append(patch.holomorphic_volume_density(u0))
        residuals.append(abs(patch.defining_residual(u0)))
    return LocalTrainingSet(
        fs_metrics=np.stack(fs_metrics, axis=0),
        basis_hessians=np.stack(basis_hessians, axis=0),
        omega_density=np.array(omega, dtype=float),
        patch_residuals=np.array(residuals, dtype=float),
    )


def metrics_from_coefficients(training: LocalTrainingSet, coefficients: np.ndarray) -> np.ndarray:
    return training.fs_metrics + np.einsum("a,najk->njk", coefficients, training.basis_hessians)


def objective_from_coefficients(
    coefficients: np.ndarray,
    training: LocalTrainingSet,
    positivity_weight: float = 10.0,
    l2_weight: float = 1e-3,
) -> float:
    metrics = metrics_from_coefficients(training, coefficients)
    eigvals = np.linalg.eigvalsh(metrics)
    min_eigs = np.min(eigvals, axis=1)
    positive = np.all(eigvals > 0.0, axis=1)
    positivity_penalty = float(np.mean(np.maximum(1e-7 - min_eigs, 0.0) ** 2))
    if not np.any(positive):
        ma_loss = 1e3
    else:
        logdet = np.sum(np.log(eigvals[positive]), axis=1)
        raw = logdet - np.log(training.omega_density[positive])
        residual = raw - np.mean(raw)
        ma_loss = float(np.mean(residual**2))
    return ma_loss + positivity_weight * positivity_penalty + l2_weight * float(np.sum(coefficients**2))


def train_invariant_potential(
    z_train: np.ndarray,
    z_val: np.ndarray,
    cy: FermatHypersurface,
    basis_names: tuple[str, ...] = DEFAULT_BASIS_NAMES,
    max_train_points: int = 36,
    max_val_points: int = 48,
    step: float = 1e-4,
    seed: int = 0,
    maxiter: int = 80,
    positivity_weight: float = 10.0,
    l2_weight: float = 1e-3,
) -> tuple[InvariantPotentialModel, dict]:
    train_set = build_local_training_set(z_train, cy, basis_names, max_points=max_train_points, step=step)
    val_set = build_local_training_set(z_val, cy, basis_names, max_points=max_val_points, step=step)
    bounds = [(-0.75, 0.75)] * len(basis_names)
    history: list[dict] = []

    def objective(c: np.ndarray) -> float:
        return objective_from_coefficients(c, train_set, positivity_weight=positivity_weight, l2_weight=l2_weight)

    def record(stage: str, coefficients: np.ndarray) -> None:
        train_metrics = evaluate_coefficients(train_set, coefficients)
        val_metrics = evaluate_coefficients(val_set, coefficients)
        history.append(
            {
                "step": len(history),
                "stage": stage,
                "train_objective": objective(coefficients),
                "train_ma_rmse": train_metrics["monge_ampere"]["ma_residual_rmse"],
                "validation_ma_rmse": val_metrics["monge_ampere"]["ma_residual_rmse"],
                "train_pos_fail": train_metrics["metric_positivity"]["positivity_violation_rate"],
                "validation_pos_fail": val_metrics["metric_positivity"]["positivity_violation_rate"],
                "validation_min_eigenvalue": val_metrics["metric_positivity"]["min_eigenvalue"],
            }
        )

    def de_callback(xk: np.ndarray, convergence: float) -> bool:
        del convergence
        record("differential_evolution", xk)
        return False

    de = differential_evolution(
        objective,
        bounds=bounds,
        seed=seed,
        maxiter=maxiter,
        popsize=8,
        polish=False,
        tol=1e-5,
        updating="immediate",
        callback=de_callback,
    )
    record("differential_evolution_final", de.x)
    local = minimize(
        objective,
        de.x,
        method="Nelder-Mead",
        callback=lambda xk: record("nelder_mead", xk),
        options={"maxiter": 600, "xatol": 1e-7, "fatol": 1e-8},
    )
    coefficients = np.asarray(local.x if local.fun <= de.fun else de.x, dtype=float)
    record("selected", coefficients)
    train_metrics = evaluate_coefficients(train_set, coefficients)
    val_metrics = evaluate_coefficients(val_set, coefficients)
    baseline_train = evaluate_coefficients(train_set, np.zeros_like(coefficients))
    baseline_val = evaluate_coefficients(val_set, np.zeros_like(coefficients))
    metadata = {
        "target": cy.name,
        "basis_names": list(basis_names),
        "seed": seed,
        "max_train_points": max_train_points,
        "max_val_points": max_val_points,
        "finite_difference_step": step,
        "positivity_weight": positivity_weight,
        "l2_weight": l2_weight,
        "optimizer": "differential_evolution_then_nelder_mead",
        "train_objective": objective(coefficients),
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "baseline_train_metrics": baseline_train,
        "baseline_validation_metrics": baseline_val,
        "max_train_patch_residual": float(np.max(train_set.patch_residuals)),
        "max_validation_patch_residual": float(np.max(val_set.patch_residuals)),
        "history": history,
    }
    return InvariantPotentialModel(coefficients=coefficients, basis_names=basis_names, metadata=metadata), metadata


def evaluate_coefficients(training: LocalTrainingSet, coefficients: np.ndarray) -> dict:
    metrics = metrics_from_coefficients(training, coefficients)
    return {
        "metric_positivity": hermitian_eigenvalue_stats(metrics),
        "monge_ampere": monge_ampere_residual(metrics, training.omega_density),
    }
