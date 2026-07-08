from __future__ import annotations

import numpy as np

from cy_expansion_lab.cy.fermat import FermatHypersurface
from cy_expansion_lab.cy.sampling import sample_fermat
from cy_expansion_lab.fit.k3_potential import (
    build_local_training_set,
    evaluate_coefficients,
    objective_from_coefficients,
    train_invariant_potential,
)
from cy_expansion_lab.models.invariant_potential import InvariantPotentialModel, invariant_basis


def test_invariant_potential_serialization_roundtrip(tmp_path) -> None:
    model = InvariantPotentialModel(coefficients=np.array([0.1, -0.2]), basis_names=("p2", "p3"))
    path = tmp_path / "model.json"
    model.save(path)
    loaded = InvariantPotentialModel.load(path)
    assert loaded.basis_names == ("p2", "p3")
    assert np.allclose(loaded.coefficients, [0.1, -0.2])


def test_invariant_basis_is_phase_invariant() -> None:
    cy = FermatHypersurface.quartic_k3()
    z = sample_fermat(cy, n_samples=8, seed=20, include_orbits=False).z
    phase = np.exp(2j * np.pi / cy.degree)
    z2 = z.copy()
    z2[:, 0] *= phase
    assert np.allclose(invariant_basis(z), invariant_basis(z2))


def test_theory_inspired_basis_terms_are_finite() -> None:
    z = np.array([[1.0, 1.2j, -0.7, 0.4j, 0.9]], dtype=complex)
    basis = invariant_basis(z, ("centered_l5", "centered_l6", "sqrt_centered_l2"))
    assert basis.shape == (1, 3)
    assert np.all(np.isfinite(basis))
    assert basis[0, 2] >= 0.0


def test_training_objective_and_short_training_run() -> None:
    cy = FermatHypersurface.quartic_k3()
    samples = sample_fermat(cy, n_samples=24, seed=21, include_orbits=False)
    training = build_local_training_set(samples.z, cy, basis_names=("p2", "p3"), max_points=5)
    zero = np.zeros(2)
    assert np.isfinite(objective_from_coefficients(zero, training))
    assert evaluate_coefficients(training, zero)["monge_ampere"]["valid_count"] == 5
    model, metadata = train_invariant_potential(
        samples.z[:12],
        samples.z[12:],
        cy,
        basis_names=("p2", "p3"),
        max_train_points=5,
        max_val_points=5,
        seed=22,
        maxiter=1,
    )
    assert model.coefficients.shape == (2,)
    assert metadata["validation_metrics"]["monge_ampere"]["valid_count"] > 0
