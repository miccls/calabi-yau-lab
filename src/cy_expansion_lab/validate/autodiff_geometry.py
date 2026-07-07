from __future__ import annotations

from collections.abc import Callable

import numpy as np

from cy_expansion_lab.cy.autodiff import (
    JaxPotential,
    complex_hessian_autodiff,
    fs_patch_potential_jax,
    invariant_model_patch_potential_jax,
    scalar_curvature_autodiff,
)
from cy_expansion_lab.cy.fermat import FermatHypersurface
from cy_expansion_lab.cy.local_patch import FermatLocalPatch
from cy_expansion_lab.models.invariant_potential import InvariantPotentialModel
from cy_expansion_lab.validate.benchmark_metrics import summarize_log_ma


def fs_autodiff_benchmark(
    z: np.ndarray,
    strata: np.ndarray,
    cy: FermatHypersurface,
    max_points: int = 48,
    ricci_points: int = 0,
) -> dict:
    return autodiff_geometry_benchmark(
        z=z,
        strata=strata,
        cy=cy,
        potential_factory=fs_patch_potential_jax,
        model_name=f"{cy.name}_fubini_study",
        max_points=max_points,
        ricci_points=ricci_points,
    )


def invariant_model_autodiff_benchmark(
    z: np.ndarray,
    strata: np.ndarray,
    cy: FermatHypersurface,
    model: InvariantPotentialModel,
    model_name: str,
    max_points: int = 48,
    ricci_points: int = 0,
) -> dict:
    return autodiff_geometry_benchmark(
        z=z,
        strata=strata,
        cy=cy,
        potential_factory=lambda patch: invariant_model_patch_potential_jax(patch, model),
        model_name=model_name,
        max_points=max_points,
        ricci_points=ricci_points,
    )


def autodiff_geometry_benchmark(
    z: np.ndarray,
    strata: np.ndarray,
    cy: FermatHypersurface,
    potential_factory: Callable[[FermatLocalPatch], JaxPotential],
    model_name: str,
    max_points: int = 48,
    ricci_points: int = 0,
) -> dict:
    indices = _diagnostic_indices(len(z), max_points)
    records = []
    skipped = 0
    ricci_done = 0
    for idx in indices:
        try:
            patch, u0 = FermatLocalPatch.from_point(cy, z[idx])
            potential = potential_factory(patch)
            metric = complex_hessian_autodiff(potential, u0)
            eigvals = np.linalg.eigvalsh(metric)
            min_eig = float(np.min(eigvals))
            omega = patch.holomorphic_volume_density(u0)
            patch_residual = float(abs(patch.defining_residual(u0)))
            logdet = None
            raw_log_ma = None
            volume_ratio = None
            if np.all(eigvals > 0.0):
                logdet = float(np.sum(np.log(eigvals)))
                raw_log_ma = logdet - float(np.log(omega))
                volume_ratio = float(np.exp(np.clip(raw_log_ma, -700.0, 700.0)))
            ricci_scalar = None
            if ricci_done < ricci_points and np.all(eigvals > 0.0):
                ricci_scalar = scalar_curvature_autodiff(potential, u0)
                ricci_done += 1
            records.append(
                {
                    "model": model_name,
                    "target": cy.name,
                    "index": int(idx),
                    "stratum": str(strata[idx]),
                    "gauge_index": int(patch.gauge_index),
                    "eliminate_index": int(patch.eliminate_index),
                    "min_eigenvalue": min_eig,
                    "positive": bool(np.all(eigvals > 0.0)),
                    "logdet": logdet,
                    "omega_density": float(omega),
                    "raw_log_ma": raw_log_ma,
                    "volume_ratio": volume_ratio,
                    "ricci_scalar": ricci_scalar,
                    "patch_residual": patch_residual,
                    "autodiff_backend": "jax",
                }
            )
        except (FloatingPointError, ValueError, OverflowError) as exc:
            skipped += 1
            records.append(
                {
                    "model": model_name,
                    "target": cy.name,
                    "index": int(idx),
                    "stratum": str(strata[idx]),
                    "skipped": True,
                    "skip_reason": type(exc).__name__,
                    "autodiff_backend": "jax",
                }
            )
    raw_log_ma = np.array([r.get("raw_log_ma", np.nan) for r in records], dtype=float)
    min_eigs = np.array([r.get("min_eigenvalue", np.nan) for r in records], dtype=float)
    ricci = np.array([r.get("ricci_scalar", np.nan) for r in records], dtype=float)
    summary = summarize_log_ma(raw_log_ma, min_eigs, ricci_scalars=ricci)
    if np.any(np.isfinite(raw_log_ma)):
        center = float(np.nanmean(raw_log_ma[np.isfinite(raw_log_ma)]))
        for record in records:
            raw = record.get("raw_log_ma")
            record["centered_log_ma_residual"] = float(raw - center) if raw is not None else None
    summary.update(
        {
            "model": model_name,
            "target": cy.name,
            "sampled_count": int(len(indices)),
            "skipped_count": int(skipped),
            "ricci_requested_count": int(ricci_points),
            "ricci_evaluated_count": int(ricci_done),
            "autodiff_backend": "jax",
            "metric_normalization": "unweighted sample averages over valid positive local patches; volume ratio scale fitted per evaluated model/sample set",
        }
    )
    return {"summary": summary, "records": records}


def _diagnostic_indices(n: int, max_points: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    return np.linspace(0, n - 1, max_points, dtype=int)
