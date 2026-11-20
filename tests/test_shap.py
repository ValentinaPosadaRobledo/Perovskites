"""Tests del análisis SHAP de F4.3 (`evaluation.shap_analysis`), con datos sintéticos."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from perovskite_pu.evaluation.shap_analysis import (
    bagging_shap,
    feature_groups,
    importance_tables,
    stability,
)
from perovskite_pu.training.pu_bagging import make_pu_pipeline

GROUPS = {
    "magpie": r"^MagpieData (?:minimum|maximum|range|mean|avg_dev|mode) (?P<prop>.+)$",
    "named": [{"pattern": "^missingindicator_", "group": "missing"},
              {"pattern": "^geom_t$", "group": "t"}],
}  # fmt: skip


def test_feature_groups_named_first_then_magpie():
    """Los patrones con nombre mandan; las de MAGPIE se agrupan por propiedad."""
    names = ["MagpieData mean Number", "MagpieData range Number", "geom_t",
             "missingindicator_geom_t", "MagpieData mode Electronegativity"]  # fmt: skip
    assert feature_groups(names, GROUPS) == ["Number", "Number", "t", "missing",
                                             "Electronegativity"]  # fmt: skip
    with pytest.raises(ValueError, match="sin grupo"):
        feature_groups(["otra cosa"], GROUPS)


def test_importance_tables_group_is_signed_sum():
    """El SHAP de un grupo es la suma con signo antes del valor absoluto."""
    v = pd.DataFrame({"a": [1.0, -1.0], "b": [-1.0, 1.0], "c": [0.5, 0.5]})
    feat, grp = importance_tables(v, ["g", "g", "h"], np.array([1, 0]))
    g = grp.set_index("group")
    assert g.loc["g", "mean_abs_all"] == pytest.approx(0.0)  # se cancelan dentro del grupo
    assert g.loc["h", "mean_abs_all"] == pytest.approx(0.5)
    assert g.loc["g", "n_features"] == 2
    f = feat.set_index("feature")
    assert f.loc["a", "mean_signed_P"] == 1.0 and f.loc["a", "mean_signed_U"] == -1.0
    assert feat["share_all"].sum() == pytest.approx(1.0)


def test_stability_identical_seeds_and_topk():
    """Rankings idénticos: Spearman 1, Jaccard 1 y frecuencia 1 en el top-k."""
    rows = [{"seed": s, "feature": f, "mean_abs_all": w}
            for s in range(3) for f, w in zip("abcd", [4, 3, 2, 1], strict=True)]  # fmt: skip
    pairs, per = stability(pd.DataFrame(rows), "feature", "mean_abs_all", top_k=2)
    assert len(pairs) == 3
    assert pairs["spearman"].eq(1.0).all() and pairs["jaccard_top2"].eq(1.0).all()
    per = per.set_index("feature")
    assert per.loc["a", "freq_top2"] == 1.0 and per.loc["d", "freq_top2"] == 0.0
    assert per.loc["a", "rank_mean"] == 1.0


@pytest.mark.parametrize(
    ("learner", "params", "space"),
    [("random_forest", {"n_estimators": 5, "max_depth": 3}, "probability"),
     ("gradient_boosting", {"n_estimators": 10, "max_depth": 2}, "log_odds")],
)  # fmt: skip
def test_bagging_shap_is_additive(learner, params, space):
    """Base + Σ SHAP reproduce la salida cruda media de las bolsas (con NaN imputados)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    X[:5, 3] = np.nan  # activa el indicador de dato faltante
    s = (X[:, 0] + rng.normal(0, 0.5, 60) > 0.5).astype(int)
    est = make_pu_pipeline(learner, params, {"n_bags": 4}, {"add_missing_indicator": True}, 1, 2)
    est.fit(X, s)
    values, base, out, score, got = bagging_shap(est, X, ["f0", "f1", "f2", "f3"])
    assert got == space
    assert "missingindicator_f3" in values.columns
    np.testing.assert_allclose(base + values.sum(axis=1).to_numpy(), out, atol=1e-9)
    if space == "probability":
        np.testing.assert_allclose(out, score, atol=1e-12)  # con RF, la salida es la puntuación
