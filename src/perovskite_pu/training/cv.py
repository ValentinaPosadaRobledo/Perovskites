"""Ajuste y predicción por fold, con todo el preprocesamiento dentro del estimador.

El estimador debe ser un ``sklearn.pipeline.Pipeline`` que incluya imputación, escalado
y selección de features (regla 2.1.6 de CLAUDE.md). Aquí se clona en cada fold y se
ajusta SOLO con las filas de train, de modo que ninguna estadística del preprocesamiento
puede ver los datos de test.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone

from perovskite_pu.splits.schemes import Fold


@dataclass
class FoldResult:
    """Resultado de un fold: índices de test, puntuaciones y (opcional) estimador ajustado."""

    fold_index: int
    test_idx: np.ndarray
    scores: np.ndarray
    estimator: Any | None = None


def _take(X: np.ndarray | pd.DataFrame, idx: np.ndarray) -> np.ndarray | pd.DataFrame:
    """Selecciona filas por posición en arrays de numpy o DataFrames."""
    return X.iloc[idx] if isinstance(X, pd.DataFrame) else X[idx]


def fit_predict_folds(
    estimator: BaseEstimator,
    X: np.ndarray | pd.DataFrame,
    s: np.ndarray,
    folds: Sequence[Fold],
    return_estimators: bool = False,
) -> list[FoldResult]:
    """Ajusta un clon del estimador en el train de cada fold y puntúa su test.

    Args:
        estimator: Pipeline de sklearn con ``predict_proba`` (o ``decision_function``).
        X: features de todas las composiciones del pool de entrenamiento.
        s: etiquetas PU observadas (1 = positivo, 0 = no etiquetado).
        folds: folds generados por :func:`perovskite_pu.splits.make_folds`.
        return_estimators: si True, guarda el estimador ajustado (para auditar el
            preprocesamiento en los tests).
    """
    s = np.asarray(s)
    results = []
    for i, fold in enumerate(folds):
        model = clone(estimator)  # clon nuevo: nada se arrastra entre folds
        model.fit(_take(X, fold.train_idx), s[fold.train_idx])
        X_test = _take(X, fold.test_idx)
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_test)[:, 1]
        else:
            scores = model.decision_function(X_test)
        results.append(
            FoldResult(i, fold.test_idx, np.asarray(scores), model if return_estimators else None)
        )
    return results
