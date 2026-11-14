"""Tests del bagging PU (F3.1), con datos sintéticos."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.tree import DecisionTreeClassifier

from perovskite_pu.training.pu_bagging import (
    PUBaggingClassifier,
    make_base_estimator,
    make_pu_pipeline,
)


def pu_data(n_pos_lab=60, n_hidden=60, n_neg=240, seed=0):
    """Dos gaussianas: P etiquetadas, P ocultas entre U y negativas reales en U."""
    rng = np.random.default_rng(seed)
    Xp = rng.normal(1.5, 1.0, (n_pos_lab + n_hidden, 4))
    Xn = rng.normal(-1.5, 1.0, (n_neg, 4))
    X = np.vstack([Xp, Xn])
    s = np.r_[np.ones(n_pos_lab, int), np.zeros(n_hidden + n_neg, int)]
    hidden = np.r_[np.zeros(n_pos_lab, bool), np.ones(n_hidden, bool), np.zeros(n_neg, bool)]
    return X, s, hidden


class Recorder(ClassifierMixin, BaseEstimator):
    """Clasificador que guarda qué filas ve en cada ajuste (para auditar las bolsas)."""

    seen: ClassVar[list] = []

    def __init__(self, random_state=None):
        self.random_state = random_state

    def fit(self, X, y):
        Recorder.seen.append((X[:, 0].copy(), np.asarray(y).copy()))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 0.5), np.full(len(X), 0.5)])


def test_each_bag_has_all_positives_and_k_unlabeled():
    X, s, _ = pu_data()
    X = np.column_stack([np.arange(len(s)), X])  # columna 0 = índice de fila
    Recorder.seen = []
    PUBaggingClassifier(Recorder(), n_bags=7, sampling_random_state=1, random_state=2).fit(X, s)
    assert len(Recorder.seen) == 7
    pos = set(np.flatnonzero(s == 1))
    for rows, y in Recorder.seen:
        rows = rows.astype(int)
        assert set(rows[y == 1]) == pos  # todas las P, todas como clase 1
        assert (y == 0).sum() == s.sum()  # K = n.º de positivas
        assert set(rows[y == 0]) <= set(np.flatnonzero(s == 0))  # solo U como clase 0


def test_oob_scores_only_for_unlabeled():
    X, s, _ = pu_data()
    clf = PUBaggingClassifier(DecisionTreeClassifier(max_depth=3), n_bags=20,
                              sampling_random_state=0, random_state=0).fit(X, s)  # fmt: skip
    assert np.isnan(clf.oob_scores_[s == 1]).all()
    assert np.isfinite(clf.oob_scores_[s == 0]).mean() > 0.95  # casi todas fuera de alguna bolsa


def test_hidden_positives_score_above_negatives():
    # Las P ocultas en U deben puntuar más alto que las negativas reales.
    X, s, hidden = pu_data()
    clf = PUBaggingClassifier(DecisionTreeClassifier(max_depth=3), n_bags=30,
                              sampling_random_state=0, random_state=0).fit(X, s)  # fmt: skip
    p = clf.predict_proba(X)[:, 1]
    negatives = (s == 0) & ~hidden
    assert p[hidden].mean() > p[negatives].mean() + 0.3


def test_determinism_and_stream_separation():
    X, s, _ = pu_data()
    base = make_base_estimator("random_forest", {"n_estimators": 5})

    def fit(sampling, model):
        clf = PUBaggingClassifier(base, n_bags=5, sampling_random_state=sampling,
                                  random_state=model)  # fmt: skip
        return clf.fit(X, s).predict_proba(X)[:, 1]

    np.testing.assert_array_equal(fit(1, 2), fit(1, 2))
    assert not np.array_equal(fit(1, 2), fit(3, 2))  # otras muestras de U
    assert not np.array_equal(fit(1, 2), fit(1, 4))  # otros modelos base


def test_sample_size_options_and_errors():
    X, s, _ = pu_data()
    clf = PUBaggingClassifier(DecisionTreeClassifier(), n_bags=1, unlabeled_sample_size=0.5)
    assert clf._sample_size(10, 100) == 50
    assert clf.set_params(unlabeled_sample_size=7)._sample_size(10, 100) == 7
    with pytest.raises(ValueError):
        clf.set_params(unlabeled_sample_size="mal").fit(X, s)
    with pytest.raises(ValueError):
        PUBaggingClassifier(DecisionTreeClassifier()).fit(X, np.zeros(len(s), int))
    with pytest.raises(ValueError):
        PUBaggingClassifier(DecisionTreeClassifier()).fit(X, np.full(len(s), 2))


def test_base_estimators_use_single_thread():
    assert make_base_estimator("random_forest", {}).get_params()["n_jobs"] == 1
    with pytest.raises(ValueError):
        make_base_estimator("svm", {})


@pytest.mark.parametrize("indicator", [False, True])
def test_pipeline_imputes_and_drops_constants_fitted_on_train(indicator):
    X, s, _ = pu_data()
    X = np.column_stack([X, np.ones(len(s))])  # columna constante
    X[::5, 0] = np.nan  # huecos
    pipe = make_pu_pipeline("gradient_boosting", {"n_estimators": 5, "max_depth": 2},
                            {"n_bags": 3}, {"add_missing_indicator": indicator}, 0, 0)  # fmt: skip
    pipe.fit(X, s)
    p = pipe.predict_proba(X)[:, 1]
    assert np.isfinite(p).all()
    n_out = pipe.named_steps["variance"].get_support().sum()
    assert n_out == 4 + int(indicator)  # 4 features útiles (+ indicador), sin la constante
    clone(pipe)  # compatible con sklearn (clonable)
