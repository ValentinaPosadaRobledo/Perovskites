"""Tests de las features `pre_dft` (F2.1), con fórmulas de ejemplo (sin datos descargados)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from perovskite_pu.features.pre_dft import (
    GEOMETRIC_COLUMNS,
    bartel_membership,
    build_pre_dft,
    composition_features,
    family_columns,
    geometric_features,
)
from perovskite_pu.features.tolerance import default_tables

FORMULAS = ["SrTiO3", "LaMnO3", "BiFeO3", "NaKO3", "BaZrO3", "CaTiO3"]


@pytest.fixture(scope="module")
def fake_table_s1() -> pd.DataFrame:
    """Imitación mínima de la salida de `load_bartel2019` (solo las columnas usadas)."""
    return pd.DataFrame(
        {
            "reduced_formula": ["SrTiO3", "BaZrO3", "AgBrO3"],
            "is_train": [1, 0, 1],
            "exp_label": [1.0, 1.0, -1.0],
        }
    )


@pytest.fixture(scope="module")
def built(data_cfg, fake_table_s1):
    """Features, metadatos y columnas de FORMULAS con la configuración real."""
    return build_pre_dft(FORMULAS, data_cfg["features"], fake_table_s1)


def test_group_sizes_match_matminer_presets():
    _, cols = composition_features(["SrTiO3"], ["element_property_magpie", "stoichiometry",
                                                "valence_orbital", "ion_property"])  # fmt: skip
    sizes = {g: len(c) for g, c in cols.items()}
    assert sizes == {"element_property_magpie": 132, "stoichiometry": 6,
                     "valence_orbital": 8, "ion_property": 3}  # fmt: skip


def test_unknown_group_is_rejected():
    with pytest.raises(ValueError):
        composition_features(["SrTiO3"], ["no_existe"])


def test_composition_features_have_no_missing_values(built):
    table, _, columns = built
    comp_cols = [c for c in table.columns if c not in GEOMETRIC_COLUMNS.values()]
    assert not table[comp_cols].isna().any().any()
    assert set(columns["groups"]) >= {"element_property_magpie", "geometric"}


def test_families_follow_the_config(built, data_cfg):
    _, _, columns = built
    fams = columns["families"]
    # La familia principal NO lleva τ; la de sensibilidad sí (decisión 2026-11-14).
    assert "geom_tau" not in fams["pre_dft"]
    assert {"geom_t", "geom_mu"} <= set(fams["pre_dft"])
    assert set(fams["pre_dft_tau"]) - set(fams["pre_dft"]) == {"geom_tau"}
    assert fams == family_columns(data_cfg["features"]["families"], columns["groups"])


def test_features_do_not_depend_on_the_rest_of_the_set(built, data_cfg, fake_table_s1):
    # Transformación fija por fórmula: calcularla con un subconjunto da lo mismo. Es lo que
    # permite calcular las features una sola vez, fuera del Pipeline, sin fuga.
    table, _, _ = built
    sub, _, _ = build_pre_dft(["CaTiO3", "SrTiO3"], data_cfg["features"], fake_table_s1)
    np.testing.assert_allclose(sub.to_numpy(), table.loc[sub.index].to_numpy(), equal_nan=True)


def test_geometric_nan_when_no_charge_balance():
    feats, meta = geometric_features(["NaKO3", "SrTiO3"], *default_tables())
    assert feats.loc["NaKO3"].isna().all()
    assert "equilibre la carga" in meta.loc["NaKO3", "bartel_notes"]
    assert feats.loc["SrTiO3"].notna().all()


def test_geometric_values_and_substitute_flags():
    feats, meta = geometric_features(["SrTiO3", "BiFeO3"], *default_tables())
    rA, rB, rX = meta.loc["SrTiO3", ["bartel_rA", "bartel_rB", "bartel_rX"]]
    assert feats.loc["SrTiO3", "geom_t"] == pytest.approx((rA + rX) / (math.sqrt(2) * (rB + rX)))
    assert feats.loc["SrTiO3", "geom_mu"] == pytest.approx(rB / rX)
    assert not meta.loc["SrTiO3", "r_A_substitute"]
    assert meta.loc["BiFeO3", "r_A_substitute"]  # Bi3+ sin radio con NC 12


def test_bartel_membership(fake_table_s1):
    m = bartel_membership(["SrTiO3", "LaMnO3", "BaZrO3"], fake_table_s1)
    assert m["in_bartel2019"].to_dict() == {"BaZrO3": True, "LaMnO3": False, "SrTiO3": True}
    assert m.loc["BaZrO3", "bartel2019_is_train"] == 0
    assert pd.isna(m.loc["LaMnO3", "bartel2019_exp_label"])
