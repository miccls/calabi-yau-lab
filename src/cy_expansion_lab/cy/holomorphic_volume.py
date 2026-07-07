from __future__ import annotations

import numpy as np


def fermat_holomorphic_volume_proxy(z: np.ndarray, degree: int, eps: float = 1e-12) -> np.ndarray:
    """Patchwise proxy for log |Omega|^2 using the largest defining derivative.

    For a hypersurface P=0, the residue form in a patch has a denominator
    dP/dz_j = degree * z_j^(degree-1). We use the largest derivative patch as a
    stable diagnostic proxy, not a full global Monge-Ampere residual.
    """
    deriv_abs = degree * np.abs(z) ** (degree - 1)
    return -2.0 * np.log(np.max(deriv_abs, axis=1) + eps)
