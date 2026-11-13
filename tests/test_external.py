"""Tests sin red de F1.2 (datos de Gu et al. y Bartel et al.), con réplicas sintéticas."""

from __future__ import annotations

import json
import zipfile

import pandas as pd
import pytest

from perovskite_pu.data.external import (
    gu_test_composition_leakage,
    label_conflicts,
    load_bartel2019,
    load_gu2022,
    register_external,
    safe_extract,
    to_legacy_mpid,
)


@pytest.mark.parametrize(
    ("new", "old"),
    [("mp-aaaaahtd", "mp-5229"), ("mp-aaaaagwx", "mp-4651"), ("mp-5229", "mp-5229")],
)
def test_to_legacy_mpid(new, old):
    """Conversión verificada a mano: SrTiO3 es mp-5229 en Gu y mp-aaaaahtd en MP 2026."""
    assert to_legacy_mpid(new) == old


@pytest.fixture
def gu_repo(tmp_path):
    """Réplica mínima del repositorio de Gu, con sus DOS sistemas de índices.

    El índice de cristal (`#` en prediction.csv, usado en las particiones) NO coincide con
    el índice de archivo CIF (cif_sources.json): se enlazan por el conjunto de fuentes.
    """
    root = tmp_path / "gu"
    (root / "data").mkdir(parents=True)
    pred = pd.DataFrame(
        {
            "#": [0, 1, 2, 3, 4],
            "ABC3": ["SrTiO3", "SrTiO3", "BaZrO3", "CaTiO3", "KBrO3"],
            "ICSD": [True, True, None, True, True],
            "CL score": [0.9, 0.8, 0.2, 0.95, 0.6],
            "score std": [0.1] * 5,
            "sources": ["mp-5229,1", "2", "mp-3834", "3,abc", "4"],
        }
    )
    pred.to_csv(root / "prediction.csv", index=False)
    # Índice CIF deliberadamente desordenado respecto a `#`.
    cif_sources = {
        "10": ["2"],
        "11": ["mp-5229", "1"],
        "12": ["4"],
        "13": ["mp-3834"],
        "14": ["3", "abc"],
    }
    (root / "data" / "cif_sources.json").write_text(json.dumps(cif_sources))
    split = {"PTest": [0], "PVal": [3], "PTrain": [1, 4], "NTest": [], "NVal": [], "NTrain": [2]}
    (root / "splits_Perov_All.json").write_text(json.dumps([split]))
    return root


def test_load_gu2022_links_crystal_and_cif_indices(gu_repo):
    gu = load_gu2022(gu_repo).set_index("idx")
    assert gu.loc[0, "cif_id"] == "11" and gu.loc[1, "cif_id"] == "10"
    assert gu["is_positive"].tolist() == [True, True, False, True, True]  # vacío = virtual
    assert gu.loc[0, "mp_legacy_ids"] == ["mp-5229"]
    assert (gu.loc[3, "gu_A"], gu.loc[3, "gu_B"], gu.loc[3, "gu_X"]) == ("Ca", "Ti", "O")
    assert gu.loc[0, "split"] == "PTest" and gu.loc[2, "split"] == "NTrain"


def test_load_gu2022_rejects_ambiguous_sources(gu_repo):
    cif = json.loads((gu_repo / "data" / "cif_sources.json").read_text())
    cif["99"] = ["2"]  # mismo conjunto de fuentes que el CIF 10
    (gu_repo / "data" / "cif_sources.json").write_text(json.dumps(cif))
    with pytest.raises(ValueError, match="ambiguo"):
        load_gu2022(gu_repo)


def test_gu_test_composition_leakage(gu_repo):
    """El único positivo de test (SrTiO3) tiene un polimorfo positivo en train: fuga 100 %."""
    leak = gu_test_composition_leakage(load_gu2022(gu_repo))
    assert leak["test_positives"] == 1
    assert leak["composition_in_train_or_val"] == 1
    assert leak["composition_among_train_val_positives_frac"] == pytest.approx(1.0)


def test_label_conflicts_both_types(gu_repo, tmp_path):
    table = tmp_path / "TableS1.csv"
    pd.DataFrame({"ABX3": ["KBrO3", "BaZrO3", "CaTiO3"], "exp_label": [-1, 1, 1]}).to_csv(
        table, index=False
    )
    conflicts = label_conflicts(load_bartel2019(table), load_gu2022(gu_repo))
    got = dict(zip(conflicts["reduced_formula"], conflicts["type"], strict=True))
    # KBrO3: no perovskita para Bartel, positivo para Gu. BaZrO3: perovskita, solo virtual en Gu.
    assert got == {"KBrO3": "bartel_no_gu_si", "BaZrO3": "bartel_si_gu_virtual"}


def test_safe_extract_filters_types_and_rejects_traversal(tmp_path):
    good = tmp_path / "good.zip"
    with zipfile.ZipFile(good, "w") as z:
        z.writestr("repo/a.csv", "x\n")
        z.writestr("repo/model.pkl", b"\x80\x04")  # pickle: no se extrae
    out = tmp_path / "out"
    assert safe_extract(good, out, [".csv"]) == 1
    assert (out / "repo/a.csv").exists() and not (out / "repo/model.pkl").exists()
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("../escape.csv", "x\n")
    with pytest.raises(ValueError, match="peligrosa"):
        safe_extract(bad, tmp_path / "out2", [".csv"])


def test_register_external_keeps_mp_record(tmp_path):
    manifest = tmp_path / "MANIFEST.json"
    manifest.write_text(json.dumps({"mp_database_version": "2026.09.28", "files": {"a": "h"}}))
    m = register_external(manifest, {"gu2022": {"md5": "x"}})
    assert m["mp_database_version"] == "2026.09.28" and m["files"] == {"a": "h"}
    assert m["external"]["gu2022"] == {"md5": "x"} and "retrieved_utc" in m["external"]
