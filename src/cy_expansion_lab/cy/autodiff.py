from __future__ import annotations

from collections.abc import Callable

import numpy as np

try:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
except ImportError as exc:  # pragma: no cover - exercised only in broken environments
    raise ImportError("Milestone 4 autodiff geometry requires JAX. Install project dependencies with uv.") from exc

from cy_expansion_lab.cy.local_patch import FermatLocalPatch
from cy_expansion_lab.models.invariant_potential import InvariantPotentialModel

JaxPotential = Callable[["jnp.ndarray"], "jnp.ndarray"]


def reconstruct_affine_jax(patch: FermatLocalPatch, u: "jnp.ndarray") -> "jnp.ndarray":
    """Reconstruct affine Fermat coordinates in a branch-stable JAX chart."""
    w = jnp.zeros((patch.cy.n_coords,), dtype=jnp.complex128)
    w = w.at[patch.gauge_index].set(1.0 + 0.0j)
    for value, idx in zip(u, patch.free_indices):
        w = w.at[idx].set(value)
    rhs = -sum(w[i] ** patch.cy.degree for i in range(patch.cy.n_coords) if i != patch.eliminate_index)
    eliminated = jnp.asarray(patch.root_branch, dtype=jnp.complex128) * jnp.exp(jnp.log(rhs) / patch.cy.degree)
    return w.at[patch.eliminate_index].set(eliminated)


def fs_affine_potential_jax(w: "jnp.ndarray") -> "jnp.ndarray":
    return jnp.log(jnp.sum(jnp.abs(w) ** 2))


def fs_patch_potential_jax(patch: FermatLocalPatch) -> JaxPotential:
    def potential(u: "jnp.ndarray") -> "jnp.ndarray":
        return fs_affine_potential_jax(reconstruct_affine_jax(patch, u))

    return potential


def invariant_basis_jax(z: "jnp.ndarray", basis_names: tuple[str, ...]) -> "jnp.ndarray":
    if z.ndim == 1:
        z = z[None, :]
    radii = jnp.abs(z) ** 2
    r = radii / jnp.sum(radii, axis=1, keepdims=True)
    centered = r - jnp.mean(r, axis=1, keepdims=True)
    values = []
    for name in basis_names:
        if name == "entropy":
            values.append(-jnp.sum(r * jnp.log(jnp.maximum(r, 1e-15)), axis=1))
        elif name.startswith("p"):
            values.append(jnp.sum(r ** int(name[1:]), axis=1))
        elif name.startswith("e"):
            values.append(_elementary_jax(r, int(name[1:])))
        elif name == "centered_l2":
            values.append(jnp.sum(centered**2, axis=1))
        elif name == "centered_l3":
            values.append(jnp.sum(centered**3, axis=1))
        elif name == "centered_l4":
            values.append(jnp.sum(centered**4, axis=1))
        elif name == "centered_l5":
            values.append(jnp.sum(centered**5, axis=1))
        elif name == "centered_l6":
            values.append(jnp.sum(centered**6, axis=1))
        elif name == "sqrt_centered_l2":
            values.append(jnp.sqrt(jnp.sum(centered**2, axis=1) + 1e-8))
        else:
            raise ValueError(f"Unknown invariant basis term: {name}")
    return jnp.stack(values, axis=1)


def invariant_model_patch_potential_jax(patch: FermatLocalPatch, model: InvariantPotentialModel) -> JaxPotential:
    coefficients = jnp.asarray(model.coefficients, dtype=jnp.float64)
    basis_names = tuple(model.basis_names)
    gauge_shift = float(model.gauge_shift)

    def potential(u: "jnp.ndarray") -> "jnp.ndarray":
        w = reconstruct_affine_jax(patch, u)
        basis = invariant_basis_jax(w, basis_names)[0]
        return fs_affine_potential_jax(w) + jnp.dot(basis, coefficients) + gauge_shift

    return potential


def complex_hessian_jax(potential: JaxPotential, u: "jnp.ndarray") -> "jnp.ndarray":
    """Return partial_i partialbar_j K from a real-coordinate JAX Hessian."""
    u = jnp.asarray(u, dtype=jnp.complex128)
    n = u.shape[0]
    x0 = jnp.concatenate([jnp.real(u), jnp.imag(u)])

    def real_func(x: "jnp.ndarray") -> "jnp.ndarray":
        point = x[:n] + 1j * x[n:]
        return jnp.real(potential(point))

    hessian = jax.hessian(real_func)(x0)
    h_xx = hessian[:n, :n]
    h_xy = hessian[:n, n:]
    h_yx = hessian[n:, :n]
    h_yy = hessian[n:, n:]
    metric = 0.25 * (h_xx + h_yy + 1j * (h_xy - h_yx))
    return 0.5 * (metric + jnp.conjugate(metric.T))


def complex_hessian_autodiff(potential: JaxPotential, u: np.ndarray) -> np.ndarray:
    return np.asarray(complex_hessian_jax(potential, jnp.asarray(u, dtype=jnp.complex128)))


def real_gradient_autodiff(potential: JaxPotential, u: np.ndarray) -> np.ndarray:
    u_jax = jnp.asarray(u, dtype=jnp.complex128)
    n = u_jax.shape[0]
    x0 = jnp.concatenate([jnp.real(u_jax), jnp.imag(u_jax)])

    def real_func(x: "jnp.ndarray") -> "jnp.ndarray":
        point = x[:n] + 1j * x[n:]
        return jnp.real(potential(point))

    return np.asarray(jax.grad(real_func)(x0))


def logdet_metric_jax(potential: JaxPotential, u: "jnp.ndarray") -> "jnp.ndarray":
    metric = complex_hessian_jax(potential, u)
    sign, logdet = jnp.linalg.slogdet(metric)
    return jnp.real(logdet + jnp.log(sign))


def ricci_tensor_jax(potential: JaxPotential, u: "jnp.ndarray") -> "jnp.ndarray":
    return -complex_hessian_jax(lambda v: logdet_metric_jax(potential, v), u)


def ricci_tensor_autodiff(potential: JaxPotential, u: np.ndarray) -> np.ndarray:
    return np.asarray(ricci_tensor_jax(potential, jnp.asarray(u, dtype=jnp.complex128)))


def scalar_curvature_autodiff(potential: JaxPotential, u: np.ndarray) -> float:
    u_jax = jnp.asarray(u, dtype=jnp.complex128)
    metric = complex_hessian_jax(potential, u_jax)
    ricci = ricci_tensor_jax(potential, u_jax)
    scalar = jnp.real(jnp.trace(jnp.linalg.solve(metric, ricci)))
    return float(scalar)


def _elementary_jax(r: "jnp.ndarray", degree: int) -> "jnp.ndarray":
    out = jnp.zeros((r.shape[0], degree + 1), dtype=jnp.float64)
    out = out.at[:, 0].set(1.0)
    for j in range(r.shape[1]):
        for k in range(min(j + 1, degree), 0, -1):
            out = out.at[:, k].add(out[:, k - 1] * r[:, j])
    return out[:, degree]
