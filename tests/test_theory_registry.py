from __future__ import annotations

from cy_expansion_lab.research.theory_registry import theory_registry_rows


def test_mirjanic_mishra_entries_are_registered() -> None:
    rows = theory_registry_rows()
    identifiers = {row["identifier"] for row in rows}
    assert "MM2025-Prop3.3" in identifiers
    assert "MM2025-Conj6.3" in identifiers
    assert "MM2025-Prop6.4" in identifiers
    assert "MM2025-Prop6.5" in identifiers
    assert "MM2025-Conj6.6" in identifiers
    assert "MM2025-Prop6.7" in identifiers
    for row in rows:
        assert row["source"]
        assert row["url"].startswith("https://arxiv.org/")
        assert row["status"]
        assert row["implementation_status"]
        assert row["project_use"]
