"""Tests sin red de las funciones puras de la descarga de MP (F1.1)."""

from __future__ import annotations

import gzip
import json

import pytest

from perovskite_pu.data.download import build_manifest, flag_gnome, index_table, is_abo3_oxide


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        ("SrTiO3", True),
        ("Ba2Ti2O6", True),  # se reduce a BaTiO3
        ("LaAlO3", True),
        ("KNO3", True),  # estequiometría ABO3: el polianión se excluye en F1.3, no aquí
        ("Cs3BrO", False),  # ABC3 anónima con el O de subíndice 1
        ("Li2TiO3", False),  # A2BO3
        ("TiO2", False),  # binario
        ("SrTi2O5", False),
        ("SrTiO2N", False),  # cuaternario
        ("no-es-formula", False),
    ],
)
def test_is_abo3_oxide(formula, expected):
    assert is_abo3_oxide(formula) is expected


def test_flag_gnome_uses_both_criteria():
    docs = [
        {"material_id": "mp-1", "builder_meta": {"batch_id": "vasp6-statics"}},
        {"material_id": "mp-2"},  # solo aparece con GNoME
        {"material_id": "mp-3", "builder_meta": {"batch_id": "gnome_r2scan_statics"}},
    ]
    flagged = flag_gnome(docs, {"mp-1", "mp-3"})
    assert [d["is_gnome"] for d in flagged] == [False, True, True]
    assert "is_gnome" not in docs[0]  # no modifica la entrada original


def test_index_table_extracts_icsd_and_symmetry():
    docs = [
        {
            "material_id": "mp-5229",
            "formula_pretty": "SrTiO3",
            "theoretical": False,
            "database_IDs": {"icsd": ["icsd-1", "icsd-2"]},
            "symmetry": {"number": 221, "symbol": "Pm-3m", "crystal_system": "Cubic"},
            "energy_above_hull": 0.0,
            "is_gnome": False,
        },
        {"material_id": "mp-9", "formula_pretty": "Ba2Ti2O6", "theoretical": True},
    ]
    t = index_table(docs)
    assert list(t["material_id"]) == ["mp-5229", "mp-9"]
    row = t.iloc[0]
    assert (row["n_icsd_ids"], row["icsd_ids"], row["spacegroup_number"]) == (
        2,
        "icsd-1;icsd-2",
        221,
    )
    assert t.iloc[1]["reduced_formula"] == "BaTiO3"
    assert t.iloc[1]["n_icsd_ids"] == 0


def test_build_manifest_records_versions_and_hashes(tmp_path):
    f = tmp_path / "a.json.gz"
    with gzip.GzipFile(f, "wb", mtime=0) as gz:
        gz.write(json.dumps([{"x": 1}]).encode())
    m = build_manifest(
        [f], "2026.09.28", {"formula_anonymous": "ABC3"}, {"n": 1}, {"description": "d"}
    )
    assert m["mp_database_version"] == "2026.09.28"
    assert m["description"] == "d"
    assert set(m) >= {
        "mp_api_version",
        "emmet_core_version",
        "query_date_utc",
        "files",
        "counts",
        "git_commit",
    }
    assert len(m["files"]["a.json.gz"]) == 64
