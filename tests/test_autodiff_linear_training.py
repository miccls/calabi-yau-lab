from __future__ import annotations

import numpy as np

from cy_expansion_lab.cy.fermat import FermatHypersurface
from cy_expansion_lab.cy.sampling import sample_fermat
from cy_expansion_lab.fit.autodiff_linear import (
    LossWeights,
    build_autodiff_linear_training_set,
    evaluate_autodiff_coefficients,
    train_autodiff_linear_potential,
)


def test_autodiff_linear_training_smoke() -> None:
    cy = FermatHypersurface.quartic_k3()
    samples = sample_fermat(cy, n_samples=18, seed=601, include_orbits=False)
    basis = ("p2", "p3", "sqrt_centered_l2")
    train = build_autodiff_linear_training_set(samples.z[:9], samples.strata[:9], cy, basis, max_points=4)
    val = build_autodiff_linear_training_set(samples.z[9:], samples.strata[9:], cy, basis, max_points=4)
    weights = LossWeights(centered_log_ma=1.0, positivity=10.0, l2=1e-3)

    initial = evaluate_autodiff_coefficients(train, np.zeros(len(basis)), loss_weights=weights)
    model, metadata = train_autodiff_linear_potential(
        train,
        val,
        cy,
        loss_weights=weights,
        seed=602,
        maxiter=2,
    )
    trained = evaluate_autodiff_coefficients(train, model.coefficients, loss_weights=weights)

    assert model.coefficients.shape == (len(basis),)
    assert metadata["autodiff_backend"] == "jax"
    assert metadata["train_patch_count"] == train.count
    assert np.isfinite(trained["loss"]["total"])
    assert trained["loss"]["total"] <= initial["loss"]["total"] + 1e-5
