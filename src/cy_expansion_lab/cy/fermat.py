from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cy_expansion_lab.cy.projective import normalize_projective


@dataclass(frozen=True)
class FermatHypersurface:
    """Fermat hypersurface sum_i z_i^degree = 0 in CP^(n_ambient)."""

    name: str
    degree: int
    n_coords: int

    @property
    def complex_dim(self) -> int:
        return self.n_coords - 2

    @property
    def ambient_dim(self) -> int:
        return self.n_coords - 1

    @classmethod
    def quartic_k3(cls) -> "FermatHypersurface":
        return cls(name="fermat_quartic_k3", degree=4, n_coords=4)

    @classmethod
    def quintic_threefold(cls) -> "FermatHypersurface":
        return cls(name="fermat_quintic_threefold", degree=5, n_coords=5)

    def residual(self, z: np.ndarray) -> np.ndarray:
        return np.sum(z**self.degree, axis=1)

    def project_last_coordinate(self, leading: np.ndarray, root_branch: np.ndarray) -> np.ndarray:
        rhs = -np.sum(leading**self.degree, axis=1)
        roots = np.abs(rhs) ** (1 / self.degree) * np.exp(1j * np.angle(rhs) / self.degree)
        z_last = roots * root_branch
        return np.concatenate([leading, z_last[:, None]], axis=1)


def from_config(name: str) -> FermatHypersurface:
    if name == "fermat_quartic":
        return FermatHypersurface.quartic_k3()
    if name == "fermat_quintic":
        return FermatHypersurface.quintic_threefold()
    raise ValueError(f"Unknown Fermat target: {name}")


def fermat_phase_orbit(z: np.ndarray, degree: int, rng: np.random.Generator) -> np.ndarray:
    phases = np.exp(2j * np.pi * rng.integers(0, degree, size=z.shape) / degree)
    return normalize_projective(z * phases)


def permutation_orbit(z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.empty_like(z)
    for i in range(z.shape[0]):
        out[i] = z[i, rng.permutation(z.shape[1])]
    return normalize_projective(out)
