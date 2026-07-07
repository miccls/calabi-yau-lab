from __future__ import annotations

import numpy as np

from cy_expansion_lab.models.invariant_potential import invariant_basis
from cy_expansion_lab.reporting.milestone3 import LITERATURE_COMPARISONS, METRIC_DEFINITIONS, write_csv


def test_metric_definitions_state_units_and_conventions() -> None:
    assert any(row["metric"] == "centered_log_ma_rmse" for row in METRIC_DEFINITIONS)
    for row in METRIC_DEFINITIONS:
        assert row["formula"]
        assert row["unit"]
        assert row["aggregation"]
        assert "comparison" in row["comparable_to_literature"].lower() or "comparable" in row["comparable_to_literature"].lower()


def test_literature_comparisons_do_not_fake_numeric_values() -> None:
    assert len(LITERATURE_COMPARISONS) >= 3
    for row in LITERATURE_COMPARISONS:
        assert row["url"].startswith("https://")
        assert row["numeric_value"] == "not copied"
        assert "not directly" in row["unit_or_convention"]


def test_write_csv_roundtrip_shape(tmp_path) -> None:
    path = tmp_path / "rows.csv"
    write_csv(path, [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}])
    text = path.read_text(encoding="utf-8")
    assert text.splitlines()[0] == "a,b"
    assert len(text.splitlines()) == 3


def test_rich_invariant_basis_terms_are_finite() -> None:
    z = np.array([[1 + 0j, 2j, 1 - 1j, -0.5 + 0.25j]])
    basis = invariant_basis(z, ("p2", "p6", "e4", "centered_l4", "entropy"))
    assert basis.shape == (1, 5)
    assert np.isfinite(basis).all()
