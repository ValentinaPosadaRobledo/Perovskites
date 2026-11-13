"""Tests del análisis secundario S1: ¿los esquemas por grupos extrapolan de verdad?"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from perovskite_pu.evaluation.extrapolation import (
    group_extrapolation_report,
    interpolation_threshold,
    nearest_train_distances,
)
from perovskite_pu.splits import Fold, make_folds
from perovskite_pu.utils.seeds import rng_streams


def test_distance_is_zero_for_duplicated_point():
    """Si un punto de test tiene un gemelo exacto en train, su distancia es 0."""
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.5], [0.0, 0.0]])
    folds = [Fold(train_idx=np.array([0, 1, 2]), test_idx=np.array([3]))]
    d = nearest_train_distances(X, folds)
    assert d[3] == pytest.approx(0.0)
    assert np.isnan(d[:3]).all()  # nunca estuvieron en test


def test_scaler_is_fitted_on_train_only():
    """Un punto de test extremo no debe cambiar la escala: su distancia debe ser grande."""
    X = np.array([[0.0], [1.0], [2.0], [1000.0]])
    folds = [Fold(train_idx=np.array([0, 1, 2]), test_idx=np.array([3]))]
    d = nearest_train_distances(X, folds)
    # Escala de train: media 1, DE poblacional sqrt(2/3). Distancia = (1000 - 2) / sqrt(2/3).
    assert d[3] == pytest.approx(998.0 / np.sqrt(2.0 / 3.0))


def test_rejects_missing_values():
    with pytest.raises(ValueError, match="NaN"):
        nearest_train_distances(np.array([[np.nan], [1.0]]), [Fold(np.array([1]), np.array([0]))])


def test_group_split_extrapolates_more_than_random(toy_df, split_cfg, b_families):
    """Con features que dependen del elemento B, reservar un B aleja el test más que `random`."""
    rng = np.random.default_rng(0)
    # Feature sintética: cada elemento B tiene su propio "centro" bien separado. Solo una
    # columna: tras estandarizar, una columna de ruido pesaría igual que la señal y la
    # ocultaría (ver test_noise_feature_dilutes_extrapolation_signal).
    centers = {b: i * 5.0 for i, b in enumerate(sorted(toy_df["B"].unique()))}
    X = (toy_df["B"].map(centers).to_numpy() + rng.normal(0, 0.1, len(toy_df)))[:, None]
    stream = rng_streams(0)["fold_assignment"]
    d_random = nearest_train_distances(
        X, make_folds(toy_df, "random", split_cfg, b_families, rng=stream)
    )
    cfg = {
        **split_cfg,
        "loeo_B": {**split_cfg["loeo_B"], "small_group_policy": "report_separately"},
    }
    d_loeo = nearest_train_distances(X, make_folds(toy_df, "loeo_B", cfg, b_families))
    assert np.nanmedian(d_loeo) > np.nanmedian(d_random)

    thr = interpolation_threshold(d_random, 0.95)
    report = group_extrapolation_report(d_loeo, toy_df["B"], toy_df["label"], thr)
    assert set(report.columns) == {
        "group",
        "n_compositions",
        "n_positives",
        "median_distance",
        "extrapolative",
    }
    assert report["n_compositions"].sum() == len(toy_df)
    assert report["extrapolative"].any()


@pytest.mark.parametrize("q", [0.0, 1.0])
def test_interpolation_threshold_rejects_invalid_quantile(q):
    with pytest.raises(ValueError):
        interpolation_threshold(np.array([1.0, 2.0]), q)


def test_interpolation_threshold_ignores_nan():
    assert interpolation_threshold(np.array([np.nan, 1.0, 2.0, 3.0]), 0.5) == pytest.approx(2.0)


def test_report_flags_groups_above_threshold():
    report = group_extrapolation_report(
        np.array([0.1, 0.2, 5.0, 6.0]),
        pd.Series(["a", "a", "b", "b"]),
        pd.Series([1, 0, 1, 1]),
        1.0,
    )
    flags = dict(zip(report["group"], report["extrapolative"], strict=True))
    assert flags == {"a": False, "b": True}
    assert report.loc[report["group"] == "b", "n_positives"].item() == 2


def test_noise_feature_dilutes_extrapolation_signal(toy_df, split_cfg, b_families):
    """Documenta la limitación: una feature de ruido, tras estandarizar, tapa la señal.

    Con solo la feature ligada a B, más grupos de `loeo_B` superan el cuantil 0,95 de
    `random` que al añadir una columna de ruido. Motiva usar PCA en S1.
    """
    rng = np.random.default_rng(0)
    centers = {b: i * 5.0 for i, b in enumerate(sorted(toy_df["B"].unique()))}
    signal = toy_df["B"].map(centers).to_numpy() + rng.normal(0, 0.1, len(toy_df))
    noisy = np.column_stack([signal, rng.normal(0, 0.1, len(toy_df))])
    cfg = {
        **split_cfg,
        "loeo_B": {**split_cfg["loeo_B"], "small_group_policy": "report_separately"},
    }
    loeo = make_folds(toy_df, "loeo_B", cfg, b_families)
    flags = {}
    for name, X in (("signal", signal[:, None]), ("noisy", noisy)):
        rnd = make_folds(
            toy_df, "random", split_cfg, b_families, rng=rng_streams(0)["fold_assignment"]
        )
        thr = interpolation_threshold(nearest_train_distances(X, rnd), 0.95)
        rep = group_extrapolation_report(
            nearest_train_distances(X, loeo), toy_df["B"], toy_df["label"], thr
        )
        flags[name] = rep["extrapolative"].mean()
    assert flags["signal"] > flags["noisy"]


def test_pca_option_runs_and_validates():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 6))
    folds = [Fold(np.arange(20), np.arange(20, 30))]
    d = nearest_train_distances(X, folds, pca_variance=0.95)
    assert np.isfinite(d[20:]).all() and np.isnan(d[:20]).all()
    with pytest.raises(ValueError):
        nearest_train_distances(X, folds, pca_variance=1.5)
