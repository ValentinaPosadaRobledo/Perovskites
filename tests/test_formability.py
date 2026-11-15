"""Tests de los conjuntos y métricas de formabilidad (F3.4), con datos sintéticos."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pymatgen.core import Composition

from perovskite_pu.evaluation.formability import (
    auc_positive_vs_negative,
    formability_sets,
    positive_rate,
)
from perovskite_pu.evaluation.formability_run import formability_metrics


@pytest.fixture
def toy():
    """Etiquetas de F1.5 y TableS1 mínimas, con un caso de cada conjunto.

    Las fórmulas de F1.5 están en la forma reducida de pymatgen (FeTiO3 → TiFeO3).
    """
    labels = pd.DataFrame({
        "reduced_formula": ["SrTiO3", "FeTiO3", "CaCO3", "LaMnO3", "BaZrO3", "MgSiO3", "XyO3"],
        "label": ["positive", "synthesized_non_perovskite", "polyanion", "ambiguous",
                  "synthesized_non_perovskite", "unlabeled", "positive"],
        "A": ["Sr", "Fe", "Ca", "La", "Ba", "Mg", "X"],
        "B": ["Ti", "Ti", "C", "Mn", "Zr", "Si", "y"],
    })  # fmt: skip
    labels["reduced_formula"] = [Composition(f).reduced_formula for f in labels["reduced_formula"]]
    s1 = pd.DataFrame({
        "ABX3": ["FeTiO3", "CaCO3", "LaMnO3", "MgSiO3", "XyO3", "NiTiO3", "KNbO3"],
        "exp_label": [-1, -1, -1, -1, -1, -1, 1],
        "is_train": [1, 1, 0, 1, 1, 0, 1],
        "A": ["Fe", "Ca", "La", "Mg", "X", "Ni", "K"],
        "B": ["Ti", "C", "Mn", "Si", "y", "Ti", "Nb"],
        "X": ["O"] * 7,
    })  # fmt: skip
    return labels, s1


def test_formability_sets_assigns_each_case(toy):
    labels, s1 = toy
    out = formability_sets(labels, s1, reserved_b={"Zr"}, pool_compositions=["SrTiO3", "MgSiO3"])
    norm = {f: Composition(f).reduced_formula for f in ["FeTiO3", "NiTiO3"]}
    kind = out.set_index("composition")["set"].rename(index={v: k for k, v in norm.items()})
    kind = kind.to_dict()
    assert kind["FeTiO3"] == "principal"  # nuestra y de Bartel
    assert kind["NiTiO3"] == "principal"  # solo de Bartel, fuera de nuestro espacio
    assert kind["CaCO3"] == "polyanion"
    assert kind["LaMnO3"] == "ambiguous"
    assert kind["BaZrO3"] == "excluded_reserved_b"  # B reservado: nunca se puntúa
    assert kind["MgSiO3"] == "conflict_unlabeled"
    assert kind["XyO3"] == "conflict_positive"
    assert "KNbO3" not in kind and "SrTiO3" not in kind  # perovskitas: no son negativos
    src = out.set_index("composition")["source"].rename(index={v: k for k, v in norm.items()})
    src = src.to_dict()
    assert src["FeTiO3"] == "both" and src["NiTiO3"] == "bartel2019"


def test_formability_sets_refuses_scored_negatives_in_pool(toy):
    labels, s1 = toy
    with pytest.raises(AssertionError):
        pool = [Composition("FeTiO3").reduced_formula]
        formability_sets(labels, s1, reserved_b=set(), pool_compositions=pool)


def test_rates_and_auc():
    assert positive_rate(np.array([1, 0, 0, 1])) == pytest.approx(0.5)
    assert np.isnan(positive_rate(np.array([])))
    assert auc_positive_vs_negative([0.9, 0.8], [0.1, 0.2]) == pytest.approx(1.0)
    assert np.isnan(auc_positive_vs_negative([], [0.1]))


def test_formability_metrics_by_variant():
    raw = pd.DataFrame({
        "variant": "family", "model": "m", "seed": 0,
        "set": ["positive", "positive", "unlabeled", "principal", "principal", "polyanion"],
        "score": [0.9, 0.4, 0.3, 0.5, 0.1, 0.8],
        "pred": [1, 0, 0, 1, 0, 1],
    })  # fmt: skip
    m = formability_metrics(raw).iloc[0]
    assert m["tpr"] == pytest.approx(0.5)
    assert m["fpr_principal"] == pytest.approx(0.5)
    assert m["fpr_polyanion"] == pytest.approx(1.0)
    assert m["auc_principal"] == pytest.approx(0.75)  # 0,9 > ambas; 0,4 > 0,1 pero < 0,5
