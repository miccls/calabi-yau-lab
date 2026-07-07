from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from cy_expansion_lab.cy.fermat import FermatHypersurface


@dataclass(frozen=True)
class FermatLocalPatch:
    """Affine local chart on a Fermat hypersurface.

    The chart fixes one projective coordinate to one, eliminates another
    coordinate with the hypersurface equation, and uses the remaining
    coordinates as local complex coordinates.
    """

    cy: FermatHypersurface
    gauge_index: int
    eliminate_index: int
    free_indices: tuple[int, ...]
    root_branch: complex

    @classmethod
    def from_point(
        cls,
        cy: FermatHypersurface,
        z: np.ndarray,
        gauge_index: int | None = None,
        eliminate_index: int | None = None,
        min_derivative: float = 1e-10,
    ) -> tuple["FermatLocalPatch", np.ndarray]:
        if z.ndim != 1 or z.shape[0] != cy.n_coords:
            raise ValueError(f"Expected one homogeneous point with {cy.n_coords} coordinates")
        if gauge_index is None:
            gauge_index = int(np.argmax(np.abs(z)))
        if abs(z[gauge_index]) < min_derivative:
            raise ValueError("Cannot build affine patch with near-zero gauge coordinate")
        affine = z / z[gauge_index]
        candidates = [i for i in range(cy.n_coords) if i != gauge_index]
        if eliminate_index is None:
            eliminate_index = max(candidates, key=lambda i: abs(affine[i]) ** (cy.degree - 1))
        if eliminate_index == gauge_index:
            raise ValueError("Gauge and eliminated coordinates must differ")
        derivative = cy.degree * affine[eliminate_index] ** (cy.degree - 1)
        if abs(derivative) < min_derivative:
            raise ValueError("Eliminated coordinate has near-zero hypersurface derivative")
        free_indices = tuple(i for i in range(cy.n_coords) if i not in (gauge_index, eliminate_index))
        rhs = -sum(affine[i] ** cy.degree for i in range(cy.n_coords) if i != eliminate_index)
        principal = _complex_root(rhs, cy.degree)
        if abs(principal) < min_derivative:
            root_branch = 1.0 + 0.0j
        else:
            root_branch = affine[eliminate_index] / principal
        patch = cls(
            cy=cy,
            gauge_index=gauge_index,
            eliminate_index=eliminate_index,
            free_indices=free_indices,
            root_branch=root_branch,
        )
        return patch, affine[list(free_indices)]

    def reconstruct_affine(self, u: np.ndarray) -> np.ndarray:
        if u.shape != (len(self.free_indices),):
            raise ValueError(f"Expected local coordinate shape {(len(self.free_indices),)}, got {u.shape}")
        w = np.zeros(self.cy.n_coords, dtype=complex)
        w[self.gauge_index] = 1.0 + 0.0j
        for value, idx in zip(u, self.free_indices):
            w[idx] = value
        rhs = -sum(w[i] ** self.cy.degree for i in range(self.cy.n_coords) if i != self.eliminate_index)
        w[self.eliminate_index] = self.root_branch * _complex_root(rhs, self.cy.degree)
        return w

    def reconstruct_homogeneous(self, u: np.ndarray) -> np.ndarray:
        return self.reconstruct_affine(u)

    def defining_residual(self, u: np.ndarray) -> complex:
        w = self.reconstruct_affine(u)
        return complex(np.sum(w**self.cy.degree))

    def holomorphic_volume_density(self, u: np.ndarray, eps: float = 1e-30) -> float:
        """Return |Omega|^2 in the local free coordinates, up to global scale."""
        w = self.reconstruct_affine(u)
        derivative = self.cy.degree * w[self.eliminate_index] ** (self.cy.degree - 1)
        return float(1.0 / max(abs(derivative) ** 2, eps))


def _complex_root(value: complex, degree: int) -> complex:
    return abs(value) ** (1 / degree) * np.exp(1j * np.angle(value) / degree)


def fs_affine_potential(w: np.ndarray) -> float:
    return float(np.log(np.sum(np.abs(w) ** 2)))


def real_hessian(func: Callable[[np.ndarray], float], x: np.ndarray, step: float = 1e-4) -> np.ndarray:
    n = len(x)
    hessian = np.empty((n, n), dtype=float)
    f0 = float(func(x))
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = step
        f_plus = float(func(x + ei))
        f_minus = float(func(x - ei))
        hessian[i, i] = (f_plus - 2.0 * f0 + f_minus) / (step**2)
        for j in range(i + 1, n):
            ej = np.zeros(n)
            ej[j] = step
            f_pp = float(func(x + ei + ej))
            f_pm = float(func(x + ei - ej))
            f_mp = float(func(x - ei + ej))
            f_mm = float(func(x - ei - ej))
            value = (f_pp - f_pm - f_mp + f_mm) / (4.0 * step**2)
            hessian[i, j] = value
            hessian[j, i] = value
    return hessian


def complex_hessian_from_potential(
    potential: Callable[[np.ndarray], float],
    u: np.ndarray,
    step: float = 1e-4,
) -> np.ndarray:
    """Compute partial_i partialbar_j K by finite differences in real coords."""
    n = len(u)
    x0 = np.concatenate([u.real, u.imag])

    def real_func(x: np.ndarray) -> float:
        point = x[:n] + 1j * x[n:]
        return float(potential(point))

    h = real_hessian(real_func, x0, step=step)
    h_xx = h[:n, :n]
    h_xy = h[:n, n:]
    h_yx = h[n:, :n]
    h_yy = h[n:, n:]
    metric = 0.25 * (h_xx + h_yy + 1j * (h_xy - h_yx))
    return 0.5 * (metric + metric.conj().T)


def fs_patch_metric(patch: FermatLocalPatch, u: np.ndarray, step: float = 1e-4) -> np.ndarray:
    return complex_hessian_from_potential(lambda v: fs_affine_potential(patch.reconstruct_affine(v)), u, step=step)
