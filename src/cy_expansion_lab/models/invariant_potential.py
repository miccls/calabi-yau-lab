from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cy_expansion_lab.cy.local_patch import fs_affine_potential
from cy_expansion_lab.cy.projective import normalized_radii


DEFAULT_BASIS_NAMES = ("p2", "p3", "p4", "e2", "e3", "centered_l2")


@dataclass
class InvariantPotentialModel:
    """Symmetry-aware Kähler potential K = K_FS + sum theta_a B_a(r)."""

    coefficients: np.ndarray
    basis_names: tuple[str, ...] = DEFAULT_BASIS_NAMES
    gauge_shift: float = 0.0
    metadata: dict | None = None

    def correction_from_homogeneous(self, z: np.ndarray) -> np.ndarray:
        basis = invariant_basis(z, self.basis_names)
        return basis @ self.coefficients + self.gauge_shift

    def potential_from_homogeneous(self, z: np.ndarray) -> np.ndarray:
        fs = np.log(np.sum(np.abs(z) ** 2, axis=1))
        return fs + self.correction_from_homogeneous(z)

    def local_potential(self, w: np.ndarray) -> float:
        return float(fs_affine_potential(w) + self.correction_from_homogeneous(w[None, :])[0])

    def save(self, path: str | Path) -> None:
        payload = {
            "model_type": "fs_plus_symmetric_invariant_correction",
            "coefficients": self.coefficients.tolist(),
            "basis_names": list(self.basis_names),
            "gauge_shift": self.gauge_shift,
            "metadata": self.metadata or {},
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "InvariantPotentialModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            coefficients=np.array(payload["coefficients"], dtype=float),
            basis_names=tuple(payload["basis_names"]),
            gauge_shift=float(payload.get("gauge_shift", 0.0)),
            metadata=payload.get("metadata", {}),
        )


def invariant_basis(z: np.ndarray, basis_names: tuple[str, ...] = DEFAULT_BASIS_NAMES) -> np.ndarray:
    r = normalized_radii(z)
    values: list[np.ndarray] = []
    centered = r - np.mean(r, axis=1, keepdims=True)
    for name in basis_names:
        if name == "entropy":
            values.append(-np.sum(r * np.log(np.maximum(r, 1e-15)), axis=1))
        elif name.startswith("p"):
            degree = int(name[1:])
            values.append(np.sum(r**degree, axis=1))
        elif name.startswith("e"):
            degree = int(name[1:])
            values.append(_elementary(r, degree))
        elif name == "centered_l2":
            values.append(np.sum(centered**2, axis=1))
        elif name == "centered_l3":
            values.append(np.sum(centered**3, axis=1))
        elif name == "centered_l4":
            values.append(np.sum(centered**4, axis=1))
        elif name == "centered_l5":
            values.append(np.sum(centered**5, axis=1))
        elif name == "centered_l6":
            values.append(np.sum(centered**6, axis=1))
        elif name == "sqrt_centered_l2":
            values.append(np.sqrt(np.sum(centered**2, axis=1) + 1e-8))
        else:
            raise ValueError(f"Unknown invariant basis term: {name}")
    return np.stack(values, axis=1)


def _elementary(r: np.ndarray, degree: int) -> np.ndarray:
    out = np.zeros((r.shape[0], degree + 1), dtype=float)
    out[:, 0] = 1.0
    for j in range(r.shape[1]):
        upper = min(j + 1, degree)
        for k in range(upper, 0, -1):
            out[:, k] += out[:, k - 1] * r[:, j]
    return out[:, degree]
