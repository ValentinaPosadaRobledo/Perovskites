"""Tests del cribado univariante de la auditoría de features (F2.3), con datos sintéticos."""

from __future__ import annotations

import numpy as np
import pandas as pd

from perovskite_pu.features.audit import audited_columns, suspicious, univariate_auc


def test_univariate_auc_ranks_and_flags():
    y = np.array([1, 1, 1, 0, 0, 0])
    X = pd.DataFrame({
        "perfect": [9.0, 8.0, 7.0, 1.0, 2.0, 3.0],    # separa P de U sin error
        "inverse": [1.0, 2.0, 3.0, 9.0, 8.0, 7.0],    # separa al revés (AUC 0)
        "noise": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
        "constant": [1.0] * 6,                        # AUC no definido
        "missing_u": [1.0, 2.0, 3.0, np.nan, np.nan, 2.5],
    })  # fmt: skip
    s = univariate_auc(X, y).set_index("feature")
    assert s.loc["perfect", "auc"] == 1.0 and s.loc["inverse", "auc"] == 0.0
    assert np.isnan(s.loc["constant", "auc"])
    # La fracción de NaN por clase se reporta aparte (la ausencia también puede separar).
    assert s.loc["missing_u", "nan_frac_U"] == 2 / 3 and s.loc["missing_u", "nan_frac_P"] == 0
    assert s.loc["missing_u", "n"] == 4
    assert set(suspicious(s.reset_index(), 0.95)["feature"]) == {"perfect", "inverse"}


def test_audited_columns_resolves_groups():
    groups = {"g1": {"columns_from": "matminer_g"}, "g2": {"columns": ["geom_t"]}}
    out = audited_columns(groups, {"matminer_g": ["a", "b"]})
    assert out == {"a": "g1", "b": "g1", "geom_t": "g2"}
