"""Tests de F1.5: etiquetado por composición, revisión manual y banderas (datos sintéticos)."""

from __future__ import annotations

import pandas as pd
import pytest

from perovskite_pu.data.labeling import (
    apply_manual_review,
    build_labels,
    label_composition,
    matches_any,
    metastable_gap,
    site_assignment,
)


def _entries(rows: list[tuple]) -> pd.DataFrame:
    """Entradas sintéticas: (material_id, fórmula, clase, experimental, A, B, ehull, oxi_amb)."""
    cols = ["material_id", "reduced_formula", "cls", "experimental", "a_element", "b_element",
            "ehull", "oxi_ambiguous"]  # fmt: skip
    df = pd.DataFrame(rows, columns=cols)
    df["cls_final"] = df["cls"]
    df["manual_reviewed"] = False
    df["dup_group"] = df["material_id"]
    df["n_icsd_ids"] = df["experimental"].astype(int)
    return df


@pytest.mark.parametrize(
    ("classes", "experimental", "expected"),
    [
        (["perovskite", "non_perovskite"], [True, True], "positive"),
        (["perovskite"], [False], "unlabeled"),  # perovskita solo teórica: U, no P
        (["non_perovskite", "hexagonal_polytype"], [True, True], "synthesized_non_perovskite"),
        (["perovskite_borderline", "non_perovskite"], [True, True], "ambiguous"),
        (["doubtful"], [True], "ambiguous"),
        (["perovskite_borderline", "perovskite"], [True, False], "ambiguous"),
        (["polyanion", "polyanion"], [True, False], "polyanion"),
    ],
)
def test_label_composition(classes, experimental, expected):
    g = pd.DataFrame({"cls_final": classes, "experimental": experimental})
    assert label_composition(g) == expected


def test_manual_review_overrides_automatic_class():
    e = _entries([("mp-1", "LaYO3", "perovskite_borderline", True, "La", "Y", 0.0, False)])
    review = pd.DataFrame({"material_id": ["mp-1"], "clasificacion_manual": ["perovskita"]})
    out = apply_manual_review(e.drop(columns=["cls_final", "manual_reviewed"]), review)
    assert out.loc[0, "cls_final"] == "perovskite" and out.loc[0, "manual_reviewed"]
    assert label_composition(out) == "positive"


def test_manual_review_rejects_unknown_labels():
    e = _entries([("mp-1", "LaYO3", "perovskite", True, "La", "Y", 0.0, False)])
    bad = pd.DataFrame({"material_id": ["mp-1"], "clasificacion_manual": ["quizás"]})
    with pytest.raises(ValueError):
        apply_manual_review(e, bad)


def test_site_assignment_prefers_experimental_perovskite():
    e = _entries(
        [
            ("mp-1", "FeBiO3", "perovskite", True, "Bi", "Fe", 0.0, False),
            ("mp-2", "FeBiO3", "perovskite", False, "Fe", "Bi", 0.3, False),
            ("mp-3", "FeBiO3", "perovskite", False, "Fe", "Bi", 0.4, False),
        ]
    )
    assert site_assignment(e) == ("Bi", "Fe")


def test_matches_any_ignores_element_order():
    """pymatgen escribe CoBiO3; la lista de configs, BiCoO3: deben coincidir."""
    assert matches_any("CoBiO3", ["BiCoO3"])
    assert not matches_any("CoBiO3", ["BiNiO3"])


def test_metastable_gap():
    # Perovskita experimental a 0,2 eV/átomo y polimorfo no perovskita en el casco: 0,2.
    e = _entries(
        [
            ("mp-1", "MgSiO3", "perovskite", True, "Mg", "Si", 0.20, False),
            ("mp-2", "MgSiO3", "non_perovskite", True, "Mg", "Si", 0.00, False),
        ]
    )
    assert metastable_gap(e) == pytest.approx(0.20)
    # Si el estado fundamental es perovskita, el gap es 0.
    e.loc[0, "ehull"], e.loc[1, "ehull"] = 0.0, 0.1
    assert metastable_gap(e) == 0.0


def test_build_labels_flags():
    e = _entries(
        [
            ("mp-1", "MgSiO3", "perovskite", True, "Mg", "Si", 0.21, False),
            ("mp-2", "MgSiO3", "non_perovskite", True, "Mg", "Si", 0.00, False),
            ("mp-3", "FeCaO3", "perovskite", True, "Ca", "Fe", 0.0, True),
            ("mp-4", "SiSrO3", "perovskite", True, "Sr", "Si", 0.0, False),
            ("mp-5", "NiBaO3", "hexagonal_polytype", True, "Ba", "Ni", 0.0, False),
        ]
    )
    flags_cfg = {
        "high_pressure_or_metastable": ["MgSiO3"],
        "ambiguous_oxidation_state": [],
        "metastable_dft_gap_ev_per_atom": 0.05,
    }
    lab = build_labels(e, flags_cfg, bartel_non_perovskite={"SiSrO3"}).set_index("reduced_formula")
    assert lab.loc["MgSiO3", "flag_high_pressure"] and lab.loc["MgSiO3", "flag_metastable_dft"]
    assert lab.loc["FeCaO3", "flag_ambiguous_oxidation"]  # todas sus perovskitas, ambiguas
    assert lab.loc["SiSrO3", "flag_bartel_conflict"]
    assert lab.loc["NiBaO3", "label"] == "synthesized_non_perovskite"
    assert lab.loc["NiBaO3", "flag_only_hexagonal"]
    assert not lab.loc["FeCaO3", "flag_metastable_dft"]
