from __future__ import annotations

from collections.abc import Callable

import numpy as np

from cy_expansion_lab.cy.fermat import FermatHypersurface
from cy_expansion_lab.cy.local_patch import (
    FermatLocalPatch,
    complex_hessian_from_potential,
    fs_affine_potential,
)
from cy_expansion_lab.invariants.quotient import invariant_table
from cy_expansion_lab.models.invariant_potential import InvariantPotentialModel
from cy_expansion_lab.validate.kahler import hermitian_eigenvalue_stats
from cy_expansion_lab.validate.monge_ampere import monge_ampere_residual


def fs_geometry_diagnostics(
    z: np.ndarray,
    cy: FermatHypersurface,
    max_points: int = 48,
    step: float = 1e-4,
) -> dict:
    return potential_geometry_diagnostics(
        z=z,
        cy=cy,
        potential_factory=lambda patch: lambda u: fs_affine_potential(patch.reconstruct_affine(u)),
        max_points=max_points,
        step=step,
    )


def model_geometry_diagnostics(
    z: np.ndarray,
    cy: FermatHypersurface,
    model: object,
    feature_min: np.ndarray,
    power_max: int,
    max_points: int = 48,
    step: float = 1e-4,
) -> dict:
    def factory(patch: FermatLocalPatch) -> Callable[[np.ndarray], float]:
        def potential(u: np.ndarray) -> float:
            homogeneous = patch.reconstruct_homogeneous(u)[None, :]
            features, _ = invariant_table(homogeneous, power_max=power_max)
            positive_features = features - feature_min[None, :] + 1e-6
            return float(model.predict(positive_features)[0])

        return potential

    return potential_geometry_diagnostics(z=z, cy=cy, potential_factory=factory, max_points=max_points, step=step)


def invariant_potential_geometry_diagnostics(
    z: np.ndarray,
    cy: FermatHypersurface,
    model: InvariantPotentialModel,
    max_points: int = 48,
    step: float = 1e-4,
) -> dict:
    def factory(patch: FermatLocalPatch) -> Callable[[np.ndarray], float]:
        def potential(u: np.ndarray) -> float:
            return model.local_potential(patch.reconstruct_affine(u))

        return potential

    return potential_geometry_diagnostics(z=z, cy=cy, potential_factory=factory, max_points=max_points, step=step)


def potential_geometry_diagnostics(
    z: np.ndarray,
    cy: FermatHypersurface,
    potential_factory: Callable[[FermatLocalPatch], Callable[[np.ndarray], float]],
    max_points: int = 48,
    step: float = 1e-4,
) -> dict:
    indices = _diagnostic_indices(len(z), max_points)
    metrics = []
    omega = []
    residuals = []
    skipped = 0
    for idx in indices:
        try:
            patch, u0 = FermatLocalPatch.from_point(cy, z[idx])
            potential = potential_factory(patch)
            metric = complex_hessian_from_potential(potential, u0, step=step)
            metrics.append(metric)
            omega.append(patch.holomorphic_volume_density(u0))
            residuals.append(abs(patch.defining_residual(u0)))
        except (FloatingPointError, ValueError, OverflowError):
            skipped += 1
    if not metrics:
        return {
            "sampled_count": int(len(indices)),
            "used_count": 0,
            "skipped_count": int(skipped),
            "max_patch_residual": float("nan"),
            "metric_positivity": {},
            "monge_ampere": {},
        }
    metric_batch = np.stack(metrics, axis=0)
    omega_arr = np.array(omega, dtype=float)
    return {
        "sampled_count": int(len(indices)),
        "used_count": int(len(metrics)),
        "skipped_count": int(skipped),
        "max_patch_residual": float(np.max(residuals)),
        "metric_positivity": hermitian_eigenvalue_stats(metric_batch),
        "monge_ampere": monge_ampere_residual(metric_batch, omega_arr),
    }


def pointwise_geometry_records(
    z: np.ndarray,
    strata: np.ndarray,
    cy: FermatHypersurface,
    potential_factory: Callable[[FermatLocalPatch], Callable[[np.ndarray], float]],
    max_points: int = 96,
    step: float = 1e-4,
) -> list[dict]:
    indices = _diagnostic_indices(len(z), max_points)
    raw_records: list[dict] = []
    raw_values = []
    valid_positions = []
    for idx in indices:
        try:
            patch, u0 = FermatLocalPatch.from_point(cy, z[idx])
            metric = complex_hessian_from_potential(potential_factory(patch), u0, step=step)
            eigvals = np.linalg.eigvalsh(metric)
            min_eig = float(np.min(eigvals))
            omega = patch.holomorphic_volume_density(u0)
            logdet = None
            raw = None
            if np.all(eigvals > 0.0):
                logdet = float(np.sum(np.log(eigvals)))
                raw = logdet - float(np.log(omega))
                raw_values.append(raw)
                valid_positions.append(len(raw_records))
            raw_records.append(
                {
                    "index": int(idx),
                    "stratum": str(strata[idx]),
                    "min_eigenvalue": min_eig,
                    "positive": bool(np.all(eigvals > 0.0)),
                    "logdet": logdet,
                    "omega_density": float(omega),
                    "raw_log_ma": raw,
                    "patch_residual": float(abs(patch.defining_residual(u0))),
                }
            )
        except (FloatingPointError, ValueError, OverflowError):
            raw_records.append(
                {
                    "index": int(idx),
                    "stratum": str(strata[idx]),
                    "skipped": True,
                }
            )
    if raw_values:
        constant = float(np.mean(raw_values))
        for pos in valid_positions:
            raw = raw_records[pos]["raw_log_ma"]
            raw_records[pos]["centered_log_ma_residual"] = float(raw - constant)
        for record in raw_records:
            if "centered_log_ma_residual" not in record:
                record["centered_log_ma_residual"] = None
    return raw_records


def _diagnostic_indices(n: int, max_points: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    return np.linspace(0, n - 1, max_points, dtype=int)
