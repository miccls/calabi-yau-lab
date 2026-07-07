from __future__ import annotations

import numpy as np

from cy_expansion_lab.cy.fermat import FermatHypersurface, fermat_phase_orbit, permutation_orbit
from cy_expansion_lab.cy.sampling import sample_fermat
from cy_expansion_lab.invariants.quotient import invariant_table


def test_invariants_are_phase_and_permutation_invariant() -> None:
    rng = np.random.default_rng(11)
    cy = FermatHypersurface.quartic_k3()
    samples = sample_fermat(cy, n_samples=12, seed=3, include_orbits=False)
    f0, _ = invariant_table(samples.z)
    f_phase, _ = invariant_table(fermat_phase_orbit(samples.z, cy.degree, rng))
    f_perm, _ = invariant_table(permutation_orbit(samples.z, rng))
    assert np.allclose(f0, f_phase)
    assert np.allclose(f0, f_perm)
