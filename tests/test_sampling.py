from __future__ import annotations

import numpy as np

from cy_expansion_lab.cy.fermat import FermatHypersurface
from cy_expansion_lab.cy.sampling import sample_fermat


def test_fermat_sampling_satisfies_hypersurface() -> None:
    cy = FermatHypersurface.quartic_k3()
    samples = sample_fermat(cy, n_samples=24, seed=7)
    assert samples.z.shape[1] == 4
    assert np.max(np.abs(cy.residual(samples.z))) < 1e-10
    assert set(samples.split).issubset({"train", "val", "test"})
