"""¿Los esquemas por grupos extrapolan de verdad? (análisis secundario S1 del preregistro).

Li et al. (2025, Commun. Mater. 6:9) mostraron que muchas pruebas "fuera de distribución" en
materiales en realidad interpolan: los datos de test caen en regiones del espacio de
features bien cubiertas por el entrenamiento. Este módulo lo cuantifica para cada esquema:

1. Para cada composición de test, la distancia a la composición de train más cercana en el
   espacio de features estandarizado. El escalado se ajusta SOLO con el train del fold,
   igual que el preprocesamiento de los modelos (regla 2.1.6 de CLAUDE.md).
2. Una referencia de "interpolación": la distribución de esas distancias bajo la partición
   `random`. Un grupo se considera extrapolativo si su mediana de distancia supera el
   cuantil de referencia (0,95 por defecto) de la distribución `random`.
3. Un resumen por grupo, que en F4 se cruza con la TPR del grupo (Spearman, descriptivo).

Limitación: al estandarizar, cada feature pesa lo mismo en la distancia, incluidas las
irrelevantes. Con muchas features (MAGPIE) las distancias se concentran y la diferencia
entre interpolar y extrapolar se diluye. Por eso el análisis principal usa el espacio PCA
que retiene el 95 % de la varianza (`pca_variance=0.95`, ajustado solo con train), y el
espacio estandarizado completo queda como sensibilidad.

No usa etiquetas: se puede calcular en F1.7, antes de entrenar ningún modelo.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from perovskite_pu.splits.schemes import Fold


def nearest_train_distances(
    X: np.ndarray, folds: Sequence[Fold], pca_variance: float | None = None
) -> np.ndarray:
    """Distancia euclídea de cada composición de test a su vecino de train más cercano.

    Args:
        X: matriz de features (n_composiciones × n_features), sin valores faltantes.
        folds: folds de un esquema; cada composición debe estar en test en exactamente uno
            (esquemas de partición completa, como en CV).
        pca_variance: si se indica (p. ej. 0.95), proyecta tras estandarizar sobre las
            componentes principales que retienen esa fracción de varianza. Escalado y PCA
            se ajustan SOLO con el train del fold.

    Returns:
        Array de longitud n con la distancia de cada composición, calculada en el fold en
        que estuvo en test. Las composiciones que nunca estuvieron en test quedan en NaN.
    """
    X = np.asarray(X, dtype=float)
    if np.isnan(X).any():
        raise ValueError("X contiene NaN: imputar antes (ajustando solo con train).")
    dist = np.full(X.shape[0], np.nan)
    if pca_variance is not None and not 0.0 < pca_variance < 1.0:
        raise ValueError("`pca_variance` debe estar en (0, 1).")
    for fold in folds:
        # Transformación ajustada SOLO con train: ninguna estadística ve el test.
        steps = [StandardScaler()]
        if pca_variance is not None:
            steps.append(PCA(n_components=pca_variance, svd_solver="full"))
        transform = make_pipeline(*steps).fit(X[fold.train_idx])
        nn = NearestNeighbors(n_neighbors=1).fit(transform.transform(X[fold.train_idx]))
        d, _ = nn.kneighbors(transform.transform(X[fold.test_idx]))
        dist[fold.test_idx] = d[:, 0]
    return dist


def interpolation_threshold(random_distances: np.ndarray, quantile: float = 0.95) -> float:
    """Cuantil de las distancias bajo `random`: por encima, un grupo se considera extrapolativo."""
    if not 0.0 < quantile < 1.0:
        raise ValueError("`quantile` debe estar en (0, 1).")
    values = np.asarray(random_distances, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        raise ValueError("No hay distancias de referencia.")
    return float(np.quantile(values, quantile))


def group_extrapolation_report(
    distances: np.ndarray, groups: pd.Series, labels: pd.Series, threshold: float
) -> pd.DataFrame:
    """Resumen por grupo: tamaño, positivos, mediana de distancia y si extrapola.

    Args:
        distances: salida de :func:`nearest_train_distances` para el esquema.
        groups: grupo de cada composición en ese esquema.
        labels: etiqueta PU (1 = positivo) de cada composición.
        threshold: salida de :func:`interpolation_threshold` (referencia `random`).
    """
    frame = pd.DataFrame(
        {
            "group": np.asarray(groups),
            "label": np.asarray(labels),
            "distance": np.asarray(distances, dtype=float),
        }
    )
    report = frame.groupby("group").agg(
        n_compositions=("label", "size"),
        n_positives=("label", "sum"),
        median_distance=("distance", "median"),
    )
    report["extrapolative"] = report["median_distance"] > threshold
    return report.reset_index().sort_values("median_distance", ascending=False, ignore_index=True)
