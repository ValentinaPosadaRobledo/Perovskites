"""Tests de la selección del holdout por grupos de elementos B (F1.8), con datos sintéticos."""

from __future__ import annotations

import pandas as pd
import pytest

from perovskite_pu.splits.holdout import select_reserved_b, split_holdout


@pytest.fixture
def pool() -> pd.DataFrame:
    """40 elementos B en 2 familias; cada B tiene 3 positivas y 5 no etiquetadas."""
    rows = []
    for i in range(40):
        b, fam = f"B{i:02d}", "fa" if i < 20 else "fb"
        for j in range(8):
            rows.append({"composition": f"A{j}{b}O3", "B": b, "family": fam, "label": int(j < 3)})
    return pd.DataFrame(rows)


def test_selection_is_deterministic_and_within_bounds(pool):
    a = select_reserved_b(pool, holdout_seed=2026)
    b = select_reserved_b(pool, holdout_seed=2026)
    assert a.reserved_b == b.reserved_b
    assert 0.10 <= a.positive_fraction <= 0.15
    assert set(a.per_family) == {"fa", "fb"} and all(a.per_family.values())  # ambas familias


def test_selection_depends_on_seed(pool):
    assert select_reserved_b(pool, 2026).reserved_b != select_reserved_b(pool, 7).reserved_b


def test_split_holdout_is_disjoint_and_complete(pool):
    sel = select_reserved_b(pool, 2026)
    holdout, train = split_holdout(pool, sel.reserved_b)
    assert set(holdout["B"]) == set(sel.reserved_b)
    assert not set(holdout["composition"]) & set(train["composition"])
    assert len(holdout) + len(train) == len(pool)


def test_selection_fails_loudly_when_bounds_impossible(pool):
    with pytest.raises(RuntimeError):
        select_reserved_b(pool, 2026, target=0.9, bounds=(0.01, 0.02), max_attempts=5)
