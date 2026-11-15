"""Tests de los esquemas `temporal` y `cluster` (decisiones de F1.7)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from perovskite_pu.splits.schemes import cluster_groups, temporal_cutoff, temporal_split


def test_temporal_cutoff_closest_to_target():
    # 20 positivas, una por año de 2001 a 2020: el 15 % (3) son 2018, 2019 y 2020.
    years = pd.Series(range(2001, 2021))
    assert temporal_cutoff(years, 0.15) == 2018
    assert temporal_cutoff(years, 0.50) == 2011


def test_temporal_split_keeps_late_positives_only_in_test():
    years = pd.Series([1990, 2010, 2015, np.nan, np.nan, np.nan, np.nan])
    labels = pd.Series([1, 1, 1, 0, 0, 0, 0])
    (fold,) = temporal_split(years, labels, 2010, 0.5, np.random.default_rng(0))
    test, train = set(fold.test_idx), set(fold.train_idx)
    assert {1, 2} <= test and 0 in train  # positivas ≥ 2010 al test; la de 1990 a train
    assert len(test & {3, 4, 5, 6}) == 2  # la mitad de U va al test
    assert not test & train


def test_temporal_split_u_part_depends_on_rng():
    years = pd.Series([2015] + [np.nan] * 20)
    labels = pd.Series([1] + [0] * 20)
    a = temporal_split(years, labels, 2010, 0.3, np.random.default_rng(1))[0].test_idx
    b = temporal_split(years, labels, 2010, 0.3, np.random.default_rng(2))[0].test_idx
    assert len(a) == len(b) == 7 and 0 in a and set(a) != set(b)


def test_cluster_groups_are_scale_invariant():
    """Como se estandariza antes de k-means, cambiar la escala de una feature no cambia nada."""
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (30, 2)), rng.normal(5, 1, (30, 2))])
    X_scaled = X * np.array([1000.0, 1.0])
    g1 = cluster_groups(X, 2, np.random.default_rng(3))
    g2 = cluster_groups(X_scaled, 2, np.random.default_rng(3))
    # Misma partición, aunque las etiquetas de clúster puedan intercambiarse.
    assert pd.crosstab(g1, g2).gt(0).sum(axis=1).eq(1).all()


@pytest.mark.parametrize("frac", [0.0, 1.0])
def test_temporal_split_u_fraction_extremes(frac):
    years = pd.Series([2015, np.nan, np.nan])
    labels = pd.Series([1, 0, 0])
    (fold,) = temporal_split(years, labels, 2010, frac, np.random.default_rng(0))
    assert len(fold.test_idx) == 1 + int(2 * frac)


# ---------------------------------------------------------------------------
# Sensibilidad 13 (F3.6): ambiguas como positivas
# ---------------------------------------------------------------------------
def test_pool_with_ambiguous_as_positive():
    """Añade solo las ambiguas con B no reservado, como P, sin año; rechaza duplicados."""
    from perovskite_pu.splits.build import pool_with_ambiguous_as_positive

    fam = {"3d": ["Ti", "Fe"], "default_family": "main_group"}
    pool = pd.DataFrame({"composition": ["SrTiO3"], "A": ["Sr"], "B": ["Ti"], "label": [1],
                         "year": [1950.0], "family": ["3d"], "flag_x": [False]})  # fmt: skip
    labels = pd.DataFrame({"reduced_formula": ["BaFeO3", "BaMnO3", "CaTiO3"],
                           "label": ["ambiguous", "ambiguous", "unlabeled"],
                           "A": ["Ba", "Ba", "Ca"], "B": ["Fe", "Mn", "Ti"],
                           "flag_x": [True, False, False]})  # fmt: skip
    out = pool_with_ambiguous_as_positive(pool, labels, {"Mn"}, fam)
    assert out["composition"].tolist() == ["BaFeO3", "SrTiO3"]
    added = out.set_index("composition").loc["BaFeO3"]
    assert added["label"] == 1 and bool(added["added_ambiguous"]) and np.isnan(added["year"])
    assert added["family"] == "3d" and bool(added["flag_x"])
    dup = labels.assign(reduced_formula=["SrTiO3", "BaMnO3", "CaTiO3"])
    with pytest.raises(ValueError, match="ya presentes"):
        pool_with_ambiguous_as_positive(pool, dup, {"Mn"}, fam)
