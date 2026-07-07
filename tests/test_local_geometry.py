from __future__ import annotations

import numpy as np

from cy_expansion_lab.cy.fermat import FermatHypersurface
from cy_expansion_lab.cy.local_patch import (
    FermatLocalPatch,
    complex_hessian_from_potential,
    fs_affine_potential,
    fs_patch_metric,
)
from cy_expansion_lab.cy.sampling import sample_fermat
from cy_expansion_lab.validate.local_geometry import fs_geometry_diagnostics


def test_local_patch_reconstructs_fermat_constraint() -> None:
    cy = FermatHypersurface.quartic_k3()
    z = sample_fermat(cy, n_samples=8, seed=10, include_orbits=False).z[0]
    patch, u = FermatLocalPatch.from_point(cy, z)
    w = patch.reconstruct_affine(u)
    assert abs(np.sum(w**cy.degree)) < 1e-10
    assert abs(w[patch.gauge_index] - 1.0) < 1e-12
    assert patch.holomorphic_volume_density(u) > 0.0


def test_complex_hessian_matches_flat_quadratic() -> None:
    u = np.array([0.2 + 0.1j, -0.3 + 0.4j])
    metric = complex_hessian_from_potential(lambda v: float(np.sum(np.abs(v) ** 2)), u)
    assert np.allclose(metric, np.eye(2), atol=1e-6)


def test_fs_patch_metric_is_positive() -> None:
    cy = FermatHypersurface.quartic_k3()
    z = sample_fermat(cy, n_samples=8, seed=11, include_orbits=False).z[1]
    patch, u = FermatLocalPatch.from_point(cy, z)
    metric = fs_patch_metric(patch, u)
    eigvals = np.linalg.eigvalsh(metric)
    assert np.min(eigvals) > 0.0


def test_fs_potential_patch_transition_sanity() -> None:
    cy = FermatHypersurface.quartic_k3()
    z = sample_fermat(cy, n_samples=8, seed=12, include_orbits=False).z[2]
    gauges = np.argsort(-np.abs(z))[:2]
    values = []
    for gauge in gauges:
        patch, u = FermatLocalPatch.from_point(cy, z, gauge_index=int(gauge))
        values.append(fs_affine_potential(patch.reconstruct_affine(u)))
    # Fubini-Study potentials differ by a pluriharmonic gauge term between
    # affine charts, so a finite, non-huge difference is the relevant sanity check.
    assert np.isfinite(values).all()
    assert abs(values[0] - values[1]) < 10.0


def test_fs_geometry_diagnostics_reports_true_metrics() -> None:
    cy = FermatHypersurface.quartic_k3()
    z = sample_fermat(cy, n_samples=16, seed=13).z
    diag = fs_geometry_diagnostics(z, cy, max_points=6)
    assert diag["used_count"] == 6
    assert diag["max_patch_residual"] < 1e-8
    assert diag["metric_positivity"]["positivity_violation_rate"] == 0.0
    assert diag["monge_ampere"]["valid_count"] == 6
