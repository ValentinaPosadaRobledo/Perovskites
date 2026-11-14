"""Tests de las features `post_dft` (F2.2) y del acceso a familias, con datos sintéticos."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from perovskite_pu.features import sets
from perovskite_pu.features.post_dft import (
    THERMO_COLUMNS,
    build_post_dft,
    family_columns_post,
    perovskite_polymorph_thermo,
)

POST_CFG = {
    "perovskite_classes": ["perovskite"],
    "thermo_type": "GGA_GGA+U",
    "sensitivity_thermo_type": "R2SCAN",  # la API devuelve `r2SCAN`: se compara sin mayúsculas
}


@pytest.fixture
def entries() -> pd.DataFrame:
    """Cuatro composiciones con distintas combinaciones de polimorfos."""
    rows = [
        # AAO3: perovskita experimental y perovskita hipotética; una no perovskita más estable.
        ("mp-1", "AAO3", "perovskite", True),
        ("mp-2", "AAO3", "perovskite", False),
        ("mp-3", "AAO3", "non_perovskite", False),
        # BBO3: solo no perovskita y borderline (no cuentan como perovskita).
        ("mp-4", "BBO3", "non_perovskite", False),
        ("mp-5", "BBO3", "perovskite_borderline", False),
        # CCO3: dos perovskitas con la misma E_hull (empate) y solo r2SCAN para una de ellas.
        ("mp-7", "CCO3", "perovskite", False),
        ("mp-6", "CCO3", "perovskite", False),
        # DDO3: perovskita solo con r2SCAN (como los compuestos con Yb de MP).
        ("mp-8", "DDO3", "perovskite", False),
    ]
    return pd.DataFrame(rows, columns=["material_id", "reduced_formula", "cls_final",
                                       "experimental"])  # fmt: skip


@pytest.fixture
def thermo() -> pd.DataFrame:
    rows = [
        ("mp-1", "GGA_GGA+U", -2.0, 0.00),
        ("mp-2", "GGA_GGA+U", -1.9, 0.10),
        ("mp-3", "GGA_GGA+U", -2.1, -0.05),  # más estable, pero no es perovskita
        ("mp-4", "GGA_GGA+U", -1.5, 0.00),
        ("mp-5", "GGA_GGA+U", -1.4, 0.02),
        ("mp-6", "GGA_GGA+U", -1.0, 0.20),
        ("mp-7", "GGA_GGA+U", -1.1, 0.20),
        ("mp-6", "r2SCAN", -1.2, 0.15),
        ("mp-8", "r2SCAN", -0.8, 0.30),
    ]
    return pd.DataFrame(rows, columns=["material_id", "thermo_type", "formation_energy_per_atom",
                                       "energy_above_hull"])  # fmt: skip


def test_lowest_energy_perovskite_not_ground_state(entries, thermo):
    best = perovskite_polymorph_thermo(entries, thermo, "GGA_GGA+U", ["perovskite"])
    # AAO3: la perovskita de menor E_hull (mp-1), aunque la no perovskita mp-3 sea más estable.
    assert best.loc["AAO3", "material_id"] == "mp-1"
    assert best.loc["AAO3", "n_perovskite_entries"] == 2
    assert "BBO3" not in best.index  # borderline no cuenta


def test_ties_are_broken_by_material_id(entries, thermo):
    best = perovskite_polymorph_thermo(entries, thermo, "GGA_GGA+U", ["perovskite"])
    assert best.loc["CCO3", "material_id"] == "mp-6"


def test_build_marks_eligibility_and_keeps_nan(entries, thermo):
    table, meta = build_post_dft(["AAO3", "BBO3", "CCO3", "DDO3"], entries, thermo, POST_CFG)
    assert meta["eligible_post_dft"].to_dict() == {"AAO3": True, "BBO3": False, "CCO3": True,
                                                    "DDO3": False}  # fmt: skip
    # r2SCAN: tipo comparado sin mayúsculas. DDO3 solo tiene r2SCAN: no es elegible para la
    # sensibilidad, que compara con GGA en el mismo subconjunto. Su E_hull r2SCAN sí se guarda.
    assert meta["eligible_post_dft_r2scan"].to_dict() == {
        "AAO3": False,
        "BBO3": False,
        "CCO3": True,
        "DDO3": False,
    }
    assert table.loc["DDO3", "thermo_ehull_r2scan"] == 0.30
    assert table.loc["BBO3"].isna().all()
    assert table.loc["AAO3", "thermo_ehull"] == 0.0 and table.loc["AAO3", "thermo_eform"] == -2.0
    assert meta.loc["AAO3", "ground_state_ehull_gga"] == -0.05


def test_metadata_never_in_feature_table(entries, thermo):
    table, meta = build_post_dft(["AAO3", "CCO3"], entries, thermo, POST_CFG)
    assert set(table.columns) == set(THERMO_COLUMNS.values())
    assert not set(table.columns) & set(meta.columns)
    assert all(c.startswith("thermo_") for c in table.columns)


def test_family_columns_post_extends_pre():
    fams = {"post_dft": {"extends": "pre_dft", "thermo": ["ehull", "eform"]}}
    out = family_columns_post(fams, {"pre_dft": ["f1", "f2"]})
    assert out == {"post_dft": ["f1", "f2", "thermo_ehull", "thermo_eform"]}


@pytest.fixture
def feature_files(tmp_path, monkeypatch, entries, thermo):
    """Tablas de features mínimas en un directorio temporal, enlazadas a `sets`."""
    pre = pd.DataFrame({"f1": [1.0, 2.0, 3.0]}, index=pd.Index(["AAO3", "BBO3", "CCO3"]))
    post, meta = build_post_dft(["AAO3", "BBO3", "CCO3"], entries, thermo, POST_CFG)
    paths = {k: str(tmp_path / v) for k, v in {
        "table": "pre.parquet", "columns": "pre.json", "post_table": "post.parquet",
        "post_metadata": "post_meta.parquet", "post_columns": "post.json"}.items()}  # fmt: skip
    pre.to_parquet(paths["table"])
    post.to_parquet(paths["post_table"])
    meta.to_parquet(paths["post_metadata"])
    with open(paths["columns"], "w", encoding="utf-8") as fh:
        json.dump({"families": {"pre_dft": ["f1"]}}, fh)
    with open(paths["post_columns"], "w", encoding="utf-8") as fh:
        json.dump({"families": {"post_dft": ["f1", "thermo_ehull", "thermo_eform"]},
                   "eligibility": {"post_dft": "eligible_post_dft"}}, fh)  # fmt: skip
    # Rutas absolutas: `ROOT / ruta_absoluta` devuelve la ruta absoluta.
    monkeypatch.setattr(sets, "_output_paths", lambda: paths)
    return paths


def test_sets_loads_family_columns_in_order(feature_files):
    assert sets.families() == ["post_dft", "pre_dft"]
    X = sets.load_feature_matrix("post_dft", ["CCO3", "AAO3"])
    assert list(X.index) == ["CCO3", "AAO3"]
    assert list(X.columns) == ["f1", "thermo_ehull", "thermo_eform"]


def test_sets_refuses_ineligible_compositions(feature_files):
    # BBO3 no tiene polimorfo perovskita: no se imputa, se rechaza.
    with pytest.raises(ValueError, match="no elegibles"):
        sets.load_feature_matrix("post_dft", ["AAO3", "BBO3"])
    assert sets.eligible("post_dft", ["AAO3", "BBO3"]).tolist() == [True, False]
    assert sets.eligible("pre_dft", ["AAO3", "BBO3"]).all()
    assert sets.load_feature_matrix("pre_dft", ["BBO3"]).shape == (1, 1)


def test_sets_unknown_family(feature_files):
    with pytest.raises(KeyError):
        sets.load_feature_matrix("no_existe", ["AAO3"])


def test_only_exclude_policy_is_allowed(entries, thermo):
    with pytest.raises(ValueError, match="exclude"):
        build_post_dft(["AAO3"], entries, thermo, {**POST_CFG, "missing_policy": "impute"})
