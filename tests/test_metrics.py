"""Tests de las métricas PU sobre casos con resultado conocido calculado a mano."""

from __future__ import annotations

import math

import numpy as np
import pytest

from perovskite_pu.evaluation.metrics import (
    lee_liu_score,
    lower_bound_metrics,
    micro_tpr,
    predicted_positive_rate,
    threshold_for_ppr,
    threshold_for_recall,
    tpr_at_ppr,
    tpr_ppr_curve,
    true_positive_rate,
    unlabeled_positive_rate,
)

# Caso base: 4 positivos etiquetados y 2 no etiquetados.
#   s      = 1 1 1 1 0 0
#   y_pred = 1 1 0 0 1 0
# TPR = 2/4 = 0.5 ; P(ŷ=1) = 3/6 = 0.5 ; P(ŷ=1|U) = 1/2 ; Lee–Liu = 0.5² / 0.5 = 0.5
S = [1, 1, 1, 1, 0, 0]
Y = [1, 1, 0, 0, 1, 0]


def test_basic_case():
    assert true_positive_rate(S, Y) == pytest.approx(0.5)
    assert predicted_positive_rate(Y) == pytest.approx(0.5)
    assert unlabeled_positive_rate(S, Y) == pytest.approx(0.5)
    assert lee_liu_score(S, Y) == pytest.approx(0.5)


def test_perfect_ranking_lee_liu():
    """Clasificador que acierta exactamente los positivos: recall 1, P(ŷ=1)=1/3, Lee–Liu = 3."""
    s = [1, 1, 0, 0, 0, 0]
    assert true_positive_rate(s, s) == 1.0
    assert lee_liu_score(s, s) == pytest.approx(3.0)


def test_predict_all_positive():
    """Predecir todo positivo da TPR = 1 pero P(ŷ=1|U) = 1: por eso se reporta el control."""
    y = [1] * len(S)
    assert true_positive_rate(S, y) == 1.0
    assert unlabeled_positive_rate(S, y) == 1.0
    # Lee–Liu = 1 / 1 = 1, peor que el clasificador perfecto del test anterior.
    assert lee_liu_score(S, y) == pytest.approx(1.0)


def test_undefined_cases_return_nan():
    assert math.isnan(true_positive_rate([0, 0], [1, 0]))  # fold sin positivos
    assert math.isnan(unlabeled_positive_rate([1, 1], [1, 0]))  # sin no etiquetados
    assert math.isnan(lee_liu_score([1, 0], [0, 0]))  # P(ŷ=1) = 0 -> 0/0


def test_rejects_non_binary_inputs():
    with pytest.raises(ValueError):
        true_positive_rate([1, 2], [1, 0])
    with pytest.raises(ValueError):
        predicted_positive_rate([0.3, 0.7])


@pytest.mark.parametrize(
    ("target", "expected"),
    [(0.0, math.inf), (0.2, 0.9), (0.3, 0.8), (0.4, 0.8), (1.0, 0.5)],
)
def test_threshold_for_ppr(target, expected):
    """k = ceil(target · n) puntuaciones más altas; con n = 5: 0.3 -> k=2, 0.4 -> k=2."""
    scores = [0.5, 0.9, 0.7, 0.8, 0.6]
    assert threshold_for_ppr(scores, target) == expected


def test_threshold_for_ppr_rejects_invalid_target():
    with pytest.raises(ValueError):
        threshold_for_ppr([0.1, 0.2], 1.5)


def test_tpr_at_ppr_with_ties_reports_achieved_ppr():
    """Con empates en el umbral, la P(ŷ=1) alcanzada supera la pedida y se informa."""
    s = [1, 0, 1, 0]
    scores = [0.9, 0.8, 0.8, 0.1]
    tpr, ppr = tpr_at_ppr(s, scores, 0.5)  # k=2 -> umbral 0.8 -> 3 predichos positivos
    assert ppr == pytest.approx(0.75)
    assert tpr == pytest.approx(1.0)
    tpr, ppr = tpr_at_ppr(s, scores, 0.25)  # k=1 -> umbral 0.9
    assert (tpr, ppr) == (pytest.approx(0.5), pytest.approx(0.25))


def test_tpr_ppr_curve_endpoints_and_monotonicity():
    rng = np.random.default_rng(0)
    s = rng.integers(0, 2, 200)
    scores = rng.random(200) + 0.3 * s  # algo de señal
    thr, tpr, ppr = tpr_ppr_curve(s, scores)
    assert (tpr[0], ppr[0]) == (0.0, 0.0)
    assert (tpr[-1], ppr[-1]) == (1.0, 1.0)
    assert np.all(np.diff(tpr) >= 0) and np.all(np.diff(ppr) >= 0)
    assert np.all(np.diff(thr) < 0)


def test_micro_tpr_differs_from_macro():
    """Micro: Σ TP / Σ P = 1/3. El promedio por fold daría (0.5 + 0) / 2 = 0.25."""
    folds = [
        ([1, 1], [1, 0]),  # 1 de 2 positivos
        ([1, 0, 0], [0, 0, 1]),  # 0 de 1 positivo
        ([0], [1]),  # sin positivos: no aporta al denominador
    ]
    assert micro_tpr(folds) == pytest.approx(1 / 3)
    assert math.isnan(micro_tpr([([0, 0], [1, 1])]))


def test_lower_bound_metrics_are_labelled_as_such():
    out = lower_bound_metrics(S, Y, scores=[0.9, 0.8, 0.3, 0.2, 0.7, 0.1])
    assert set(out) == {"accuracy_lower_bound", "f1_lower_bound", "roc_auc_lower_bound"}
    # accuracy con U como negativo: aciertos en posiciones 0, 1, 5 -> 3/6
    assert out["accuracy_lower_bound"] == pytest.approx(0.5)


def test_threshold_for_recall_keeps_target_fraction_of_positives():
    """10 positivos con puntuaciones 0,1…1,0: recall 0,90 -> 9 por encima -> umbral 0,2."""
    pos_scores = [round(0.1 * i, 1) for i in range(1, 11)]
    s = [1] * 10 + [0, 0]
    scores = [*pos_scores, 0.95, 0.05]  # los no etiquetados no influyen en el umbral
    thr = threshold_for_recall(s, scores, 0.90)
    assert thr == pytest.approx(0.2)
    assert true_positive_rate(s, (np.asarray(scores) >= thr).astype(int)) == pytest.approx(0.9)


def test_threshold_for_recall_rounds_up_to_reach_target():
    """Con 7 positivos, 0,90 · 7 = 6,3 -> se exigen 7: recall alcanzado ≥ objetivo."""
    s = [1] * 7
    scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
    assert threshold_for_recall(s, scores, 0.90) == pytest.approx(0.3)


@pytest.mark.parametrize("target", [0.0, 1.5])
def test_threshold_for_recall_rejects_invalid_target(target):
    with pytest.raises(ValueError):
        threshold_for_recall([1, 0], [0.5, 0.1], target)


def test_threshold_for_recall_requires_positives():
    with pytest.raises(ValueError, match="positivos"):
        threshold_for_recall([0, 0], [0.5, 0.1])
