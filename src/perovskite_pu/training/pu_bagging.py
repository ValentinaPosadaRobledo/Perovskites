"""F3.1: bagging PU (Mordelet y Vert, 2014) con Random Forest o Gradient Boosting.

Algoritmo (para cada una de las `n_bags` bolsas):
1. Se toman TODAS las positivas etiquetadas (s = 1) y una muestra bootstrap (con reemplazo)
   de K no etiquetadas (s = 0); por defecto K = n.º de positivas, así cada bolsa está
   equilibrada.
2. Se entrena un clasificador base que separa P de esa muestra de U.
3. La puntuación de una composición nueva es la media de `predict_proba` sobre las bolsas.

Las no etiquetadas que quedan fuera de una bolsa reciben su predicción (fuera de bolsa); la
media de esas predicciones se guarda en `oob_scores_` (puntuación transductiva de U, útil para
estimar la prior π en F3.2). Las positivas están en todas las bolsas, así que no tienen
puntuación fuera de bolsa.

Semillas (regla 2.2): `sampling_random_state` controla las muestras de U (flujo `pu_sampling`)
y `random_state` los clasificadores base (flujo `model_init`). Así, cambiar el aprendiz no
altera qué no etiquetadas entran en cada bolsa. `n_jobs = 1` en los modelos base: solo se
paraleliza en el nivel semilla × esquema.

Todo el preprocesamiento va en un `Pipeline` (`make_pu_pipeline`), que se ajusta solo con el
train de cada fold (regla 2.1.6).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


class PUBaggingClassifier(ClassifierMixin, BaseEstimator):
    """Bagging PU con un clasificador base de sklearn (API de clasificador binario)."""

    def __init__(
        self,
        estimator: BaseEstimator | None = None,
        n_bags: int = 50,
        unlabeled_sample_size: int | float | str = "n_positive",
        sampling_random_state: int | None = None,
        random_state: int | None = None,
    ):
        self.estimator = estimator
        self.n_bags = n_bags
        self.unlabeled_sample_size = unlabeled_sample_size
        self.sampling_random_state = sampling_random_state
        self.random_state = random_state

    def _sample_size(self, n_pos: int, n_unl: int) -> int:
        """K: tamaño de la muestra de U en cada bolsa."""
        k = self.unlabeled_sample_size
        if k == "n_positive":
            return n_pos
        if isinstance(k, float) and 0 < k <= 1:
            return max(1, round(k * n_unl))
        if isinstance(k, int) and k > 0:
            return k
        raise ValueError(f"unlabeled_sample_size no válido: {k!r}")

    def fit(self, X, s):
        X = np.asarray(X)
        s = np.asarray(s).astype(int)
        if set(np.unique(s)) - {0, 1}:
            raise ValueError("Las etiquetas PU deben ser 0 (no etiquetada) o 1 (positiva).")
        pos, unl = np.flatnonzero(s == 1), np.flatnonzero(s == 0)
        if pos.size == 0 or unl.size == 0:
            raise ValueError("Hacen falta positivas y no etiquetadas para el bagging PU.")
        k = self._sample_size(pos.size, unl.size)
        base = self.estimator if self.estimator is not None else RandomForestClassifier()
        rng_sample = np.random.default_rng(self.sampling_random_state)
        # Una semilla por bolsa para el modelo base, derivada de `random_state`.
        model_seeds = np.random.default_rng(self.random_state).integers(0, 2**32 - 1, self.n_bags)
        oob_sum, oob_n = np.zeros(X.shape[0]), np.zeros(X.shape[0])
        self.estimators_ = []
        for b in range(self.n_bags):
            u_bag = rng_sample.choice(unl, size=k, replace=True)  # bootstrap de U
            idx = np.concatenate([pos, u_bag])
            y = np.r_[np.ones(pos.size, int), np.zeros(k, int)]
            model = clone(base)
            if "random_state" in model.get_params():
                model.set_params(random_state=int(model_seeds[b]))
            model.fit(X[idx], y)
            self.estimators_.append(model)
            out = np.setdiff1d(unl, u_bag, assume_unique=False)  # U fuera de la bolsa
            if out.size:
                oob_sum[out] += model.predict_proba(X[out])[:, 1]
                oob_n[out] += 1
        with np.errstate(invalid="ignore", divide="ignore"):
            self.oob_scores_ = np.where(oob_n > 0, oob_sum / oob_n, np.nan)
        self.oob_scores_[pos] = np.nan  # las P están en todas las bolsas
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        p = np.mean([m.predict_proba(X)[:, 1] for m in self.estimators_], axis=0)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        # Solo por compatibilidad: el umbral del proyecto se calibra aparte (recall 0,90).
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ---------------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------------
BASE_LEARNERS = ("random_forest", "gradient_boosting")


def make_base_estimator(learner: str, params: Mapping[str, Any]) -> BaseEstimator:
    """Clasificador base con `n_jobs = 1` (regla 2.2) y los hiperparámetros dados."""
    params = dict(params)
    if learner == "random_forest":
        return RandomForestClassifier(n_jobs=1, **params)
    if learner == "gradient_boosting":
        return GradientBoostingClassifier(**params)
    raise ValueError(f"Aprendiz base desconocido: {learner} (opciones: {BASE_LEARNERS})")


def make_pu_pipeline(
    learner: str,
    base_params: Mapping[str, Any],
    bagging_params: Mapping[str, Any],
    preprocessing: Mapping[str, Any],
    sampling_random_state: int,
    random_state: int,
) -> Pipeline:
    """Imputación (mediana del train) → umbral de varianza → bagging PU.

    `preprocessing["add_missing_indicator"]` añade una columna 0/1 por cada feature con
    huecos en el train (p. ej. t y μ sin balance de carga de Bartel).
    """
    return Pipeline([
        ("impute", SimpleImputer(strategy="median",
                                 add_indicator=bool(preprocessing.get("add_missing_indicator")),
                                 keep_empty_features=False)),
        ("variance", VarianceThreshold(threshold=0.0)),  # quita columnas constantes en el train
        ("pu", PUBaggingClassifier(estimator=make_base_estimator(learner, base_params),
                                   sampling_random_state=sampling_random_state,
                                   random_state=random_state, **bagging_params)),
    ])  # fmt: skip
