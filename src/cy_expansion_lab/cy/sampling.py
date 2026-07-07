from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cy_expansion_lab.cy.fermat import FermatHypersurface, fermat_phase_orbit, permutation_orbit
from cy_expansion_lab.cy.projective import normalize_projective


@dataclass(frozen=True)
class SampleSet:
    z: np.ndarray
    strata: np.ndarray
    split: np.ndarray
    metadata: dict


def _complex_normal(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def _roots_of_unity(degree: int, rng: np.random.Generator, n: int) -> np.ndarray:
    return np.exp(2j * np.pi * rng.integers(0, degree, size=n) / degree)


def _generic_points(cy: FermatHypersurface, rng: np.random.Generator, n: int) -> np.ndarray:
    leading = _complex_normal(rng, (n, cy.n_coords - 1))
    return normalize_projective(cy.project_last_coordinate(leading, _roots_of_unity(cy.degree, rng, n)))


def _near_coordinate_degenerate(cy: FermatHypersurface, rng: np.random.Generator, n: int) -> np.ndarray:
    leading = _complex_normal(rng, (n, cy.n_coords - 1))
    idx = rng.integers(0, cy.n_coords - 1, size=n)
    leading[np.arange(n), idx] *= rng.uniform(1e-4, 5e-2, size=n)
    return normalize_projective(cy.project_last_coordinate(leading, _roots_of_unity(cy.degree, rng, n)))


def _near_equimodular(cy: FermatHypersurface, rng: np.random.Generator, n: int) -> np.ndarray:
    phase = rng.uniform(0.0, 2 * np.pi, size=(n, cy.n_coords - 1))
    leading = np.exp(1j * phase) * (1.0 + 0.04 * rng.normal(size=(n, cy.n_coords - 1)))
    return normalize_projective(cy.project_last_coordinate(leading, _roots_of_unity(cy.degree, rng, n)))


def _fixed_type(cy: FermatHypersurface, rng: np.random.Generator, n: int) -> np.ndarray:
    z = _generic_points(cy, rng, n)
    # Pull points toward the permutation-symmetric radius profile while keeping phases.
    radii = np.sqrt(np.sum(np.abs(z) ** 2, axis=1, keepdims=True) / cy.n_coords)
    phases = np.exp(1j * np.angle(z))
    mixed = 0.65 * z + 0.35 * radii * phases
    leading = mixed[:, : cy.n_coords - 1]
    return normalize_projective(cy.project_last_coordinate(leading, _roots_of_unity(cy.degree, rng, n)))


def make_splits(rng: np.random.Generator, n: int, train: float, val: float) -> np.ndarray:
    u = rng.random(n)
    split = np.full(n, "test", dtype=object)
    split[u < train] = "train"
    split[(u >= train) & (u < train + val)] = "val"
    return split


def sample_fermat(
    cy: FermatHypersurface,
    n_samples: int,
    seed: int,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    include_orbits: bool = True,
) -> SampleSet:
    rng = np.random.default_rng(seed)
    strata_fns = [
        ("generic", _generic_points),
        ("coord_degenerate", _near_coordinate_degenerate),
        ("equimodular", _near_equimodular),
        ("fixed_type", _fixed_type),
    ]
    base_n = max(1, n_samples // len(strata_fns))
    parts: list[np.ndarray] = []
    labels: list[str] = []
    for name, fn in strata_fns:
        count = base_n if name != "generic" else n_samples - base_n * (len(strata_fns) - 1)
        pts = fn(cy, rng, count)
        parts.append(pts)
        labels.extend([name] * count)
    z = np.concatenate(parts, axis=0)
    strata = np.array(labels, dtype=object)
    if include_orbits:
        z_phase = fermat_phase_orbit(z, cy.degree, rng)
        z_perm = permutation_orbit(z, rng)
        z = np.concatenate([z, z_phase, z_perm], axis=0)
        strata = np.concatenate(
            [
                strata,
                np.array([f"{s}:phase_orbit" for s in strata], dtype=object),
                np.array([f"{s}:perm_orbit" for s in strata], dtype=object),
            ]
        )
    split = make_splits(rng, len(z), train_fraction, val_fraction)
    return SampleSet(
        z=z,
        strata=strata,
        split=split,
        metadata={
            "seed": seed,
            "target": cy.name,
            "degree": cy.degree,
            "n_coords": cy.n_coords,
            "include_orbits": include_orbits,
            "max_hypersurface_residual": float(np.max(np.abs(cy.residual(z)))),
        },
    )
