from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TheoryEntry:
    identifier: str
    source: str
    url: str
    section: str
    status: str
    short_name: str
    claim: str
    implementation_status: str
    project_use: str
    caution: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


MIRJANIC_MISHRA_SOURCE = "Mirjanic-Mishra, arXiv:2412.19778"
MIRJANIC_MISHRA_URL = "https://arxiv.org/abs/2412.19778"


DEFAULT_THEORY_REGISTRY = [
    TheoryEntry(
        identifier="MM2025-Prop3.3",
        source=MIRJANIC_MISHRA_SOURCE,
        url=MIRJANIC_MISHRA_URL,
        section="Proposition 3.3",
        status="proven in source under stated assumptions",
        short_name="phase_invariance_for_fermat_monomial_coordinates",
        claim=(
            "Coordinates that enter the Fermat defining polynomial through isolated Fermat monomials "
            "lead to phase-invariance restrictions on admissible invariant features."
        ),
        implementation_status="registry_entry_only_milestone4",
        project_use="Feature restriction, orbit-consistency diagnostic, and falsification test for K3/quintic models.",
        caution="Use only after checking that the sampled coordinate chart and model variables satisfy the source assumptions.",
    ),
    TheoryEntry(
        identifier="MM2025-Conj6.2",
        source=MIRJANIC_MISHRA_SOURCE,
        url=MIRJANIC_MISHRA_URL,
        section="Section 6, analytical properties of phi",
        status="conjectural in source",
        short_name="cross_dimensional_self_similarity",
        claim=(
            "The correction phi appears to have self-similar behavior across related Fermat hypersurfaces "
            "when expressed in suitable invariant coordinates."
        ),
        implementation_status="registry_entry_only_milestone4",
        project_use="K3/quintic comparison diagnostic and possible architecture-sharing prior.",
        caution="Do not impose as a hard constraint before Milestone 6 support/falsification tests.",
    ),
    TheoryEntry(
        identifier="MM2025-Conj6.3",
        source=MIRJANIC_MISHRA_SOURCE,
        url=MIRJANIC_MISHRA_URL,
        section="Conjecture 6.3",
        status="conjectural in source",
        short_name="phi_derivative_size_bounds",
        claim=(
            "First and second derivatives of phi are conjectured to satisfy simple size bounds after "
            "normalization by the invariant coordinates."
        ),
        implementation_status="registry_entry_only_milestone4",
        project_use="Derivative-bound probe and possible soft regularizer for Milestone 5.",
        caution="Bounds are numerically delicate near small invariant coordinates; require stability masks.",
    ),
    TheoryEntry(
        identifier="MM2025-Prop6.4",
        source=MIRJANIC_MISHRA_SOURCE,
        url=MIRJANIC_MISHRA_URL,
        section="Proposition 6.4",
        status="proposition in source",
        short_name="pseudo_origin_metric_formula",
        claim="The metric has a simple predicted form at the pseudo-origin locus.",
        implementation_status="registry_entry_only_milestone4",
        project_use="Locus target, validation diagnostic, and possible hard normalization for symbolic candidates.",
        caution="The project must define a compatible pseudo-origin locus before treating this as a target.",
    ),
    TheoryEntry(
        identifier="MM2025-Prop6.5",
        source=MIRJANIC_MISHRA_SOURCE,
        url=MIRJANIC_MISHRA_URL,
        section="Proposition 6.5",
        status="proposition in source under symmetry assumptions",
        short_name="equimodular_metric_formula",
        claim="The metric has a constrained predicted form on the equimodular locus.",
        implementation_status="registry_entry_only_milestone4",
        project_use="Equimodular-locus target, residual stratification, and symbolic-discovery constraint if supported.",
        caution="Use as a hard constraint only when the model and coordinate conventions match the source formula.",
    ),
    TheoryEntry(
        identifier="MM2025-Conj6.6",
        source=MIRJANIC_MISHRA_SOURCE,
        url=MIRJANIC_MISHRA_URL,
        section="Conjecture 6.6",
        status="conjectural in source",
        short_name="near_equimodular_scaled_fubini_study",
        claim="Near the equimodular locus, the Ricci-flat metric may be well approximated by a scaled Fubini-Study pullback.",
        implementation_status="registry_entry_only_milestone4",
        project_use="Warm start, auxiliary loss, and error-versus-distance diagnostic.",
        caution="Must be tested as an approximation, not assumed globally.",
    ),
    TheoryEntry(
        identifier="MM2025-Prop6.7",
        source=MIRJANIC_MISHRA_SOURCE,
        url=MIRJANIC_MISHRA_URL,
        section="Proposition 6.7",
        status="proposition/source-derived analytic baseline",
        short_name="sqrt_s2_correction_baseline",
        claim="A simple correction involving sqrt(s2) is analytically motivated as a baseline for phi.",
        implementation_status="registry_entry_only_milestone4",
        project_use="Baseline model and symbolic-regression seed for later milestones.",
        caution="Evaluate against true geometric metrics before using it to restrict the hypothesis space.",
    ),
]


def theory_registry_rows() -> list[dict[str, str]]:
    return [entry.as_dict() for entry in DEFAULT_THEORY_REGISTRY]


def write_theory_registry(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(theory_registry_rows(), indent=2), encoding="utf-8")
