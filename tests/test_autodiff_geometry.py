from __future__ import annotations

import numpy as np

from cy_expansion_lab.cy.autodiff import complex_hessian_autodiff, fs_patch_potential_jax
from cy_expansion_lab.cy.fermat import FermatHypersurface
from cy_expansion_lab.cy.local_patch import FermatLocalPatch, fs_patch_metric
from cy_expansion_lab.cy.sampling import sample_fermat
from cy_expansion_lab.validate.autodiff_geometry import fs_autodiff_benchmark
from cy_expansion_lab.validate.benchmark_metrics import metric_definition_rows


def test_autodiff_fs_hessian_matches_finite_difference_on_k3_sample() -> None:
    cy = FermatHypersurface.quartic_k3()
    samples = sample_fermat(cy, n_samples=8, seed=123, include_orbits=False)
    patch, u0 = FermatLocalPatch.from_point(cy, samples.z[0])

    fd_metric = fs_patch_metric(patch, u0, step=5e-5)
    ad_metric = complex_hessian_autodiff(fs_patch_potential_jax(patch), u0)

    np.testing.assert_allclose(ad_metric, fd_metric, rtol=3e-4, atol=3e-4)


def test_autodiff_benchmark_reports_metric_units() -> None:
    cy = FermatHypersurface.quartic_k3()
    samples = sample_fermat(cy, n_samples=12, seed=456, include_orbits=False)
    result = fs_autodiff_benchmark(samples.z, samples.strata, cy=cy, max_points=4, ricci_points=1)

    summary = result["summary"]
    assert summary["autodiff_backend"] == "jax"
    assert summary["valid_positive_count"] > 0
    assert summary["centered_log_ma_rmse"] is not None
    assert summary["sigma_l1_volume_ratio"] is not None
    assert summary["ricci_evaluated_count"] == 1
    assert "unweighted sample averages" in summary["metric_normalization"]


def test_metric_definitions_include_required_published_conventions() -> None:
    names = {row["metric"] for row in metric_definition_rows()}
    assert "centered_log_ma_rmse" in names
    assert "raw_volume_ratio_mae" in names
    assert "sigma_l1_volume_ratio" in names
    assert "ricci_scalar_abs_mean" in names
    for row in metric_definition_rows():
        assert row["unit"]
        assert row["aggregation"]
        assert row["convention"]
