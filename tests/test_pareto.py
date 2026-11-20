"""Tests de F5.2 (`perovskite_pu.pourbaix.pareto`), con datos sintéticos."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from perovskite_pu.pourbaix.pareto import (
    condition_front,
    oof_scores,
    pareto_ranks,
    toxicity_flags,
)

SPECIES = {"soluble_Ba2+": r"Ba\[\+2\]", "Cr6+": r"(?:H?CrO4|Cr2O7)\["}


def test_pareto_ranks_successive_fronts():
    """Maximizar x y minimizar y: frente, segundo frente y empates en el mismo rango."""
    x = np.array([0.9, 0.5, 0.8, 0.4, 0.8])
    y = np.array([0.5, 0.1, 0.6, 0.7, 0.6])
    # (0,9; 0,5) y (0,5; 0,1) no están dominados; (0,8; 0,6) dos veces, dominado por el 1.º.
    assert pareto_ranks(x, y).tolist() == [1, 1, 2, 3, 2]


def test_toxicity_flags_elements_and_species():
    """Elementos por presencia; Ba²⁺ y cromato por especie estable en la ventana."""
    t = pd.DataFrame({"A": ["Pb", "Ba", "Sr", "La"], "B": ["Ti", "Ti", "Cr", "Al"],
                      "stable_in_window": ["TiO2(s) + Pb[+2]", "Ba[+2] + TiO2(s)",
                                           "Sr[+2] + CrO4[-2]", "LaAlO3(s)"]})  # fmt: skip
    out = toxicity_flags(t, ["Pb", "Co", "Ni", "Cd", "Tl"], SPECIES).tolist()
    assert out == ["Pb", "soluble_Ba2+", "Cr6+", ""]


def test_oof_scores_requires_one_score_per_seed():
    """Media y fracción sobre el umbral; un duplicado en una semilla es un error."""
    raw = pd.DataFrame({"baseline": "pu", "scheme": "random", "seed": [0, 1, 0, 1],
                        "composition": ["X", "X", "Y", "Y"], "score": [0.2, 0.4, 0.9, 0.7],
                        "pred": [0, 1, 1, 1]})  # fmt: skip
    s = oof_scores(raw, "pu", "random").set_index("composition")
    assert s.loc["X", "score_mean"] == pytest.approx(0.3)
    assert s.loc["X", "frac_above_threshold"] == 0.5 and s.loc["Y", "n_seeds"] == 2
    with pytest.raises(ValueError, match="una vez por semilla"):
        oof_scores(pd.concat([raw, raw.iloc[[0]]]), "pu", "random")


def test_condition_front_excludes_positives_and_toxic():
    """Solo las U sin marcas reciben rango; P y marcadas quedan en 0 y fuera del frente."""
    t = pd.DataFrame({"composition": ["P1", "U1", "U2", "U3"], "label": [1, 0, 0, 0],
                      "A": ["Sr", "Ca", "Pb", "La"], "B": ["Ti", "Ti", "Ti", "Al"],
                      "pH": 7.4, "ion_conc_M": 1e-6, "score_mean": [0.99, 0.6, 0.95, 0.4],
                      "dg_max": [0.0, 0.2, 0.0, 0.1],
                      "stable_in_window": ["", "", "", ""]})  # fmt: skip
    out = condition_front(t, 7.4, 1e-6, ["Pb"], SPECIES).set_index("composition")
    assert out.loc["P1", "pareto_rank"] == 0 and out.loc["U2", "pareto_rank"] == 0
    assert out.loc["U2", "toxicity"] == "Pb"
    assert out.loc["U1", "on_front"] and out.loc["U3", "on_front"]
