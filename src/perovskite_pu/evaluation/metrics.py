"""Métricas para aprendizaje positivo–no etiquetado (sección 2.3 de CLAUDE.md).

Notación:
    s       etiqueta observada: 1 = positivo etiquetado, 0 = no etiquetado (U).
            Los no etiquetados NO son negativos.
    y_pred  predicción binaria del modelo (1 = predicho sintetizable).
    scores  puntuación continua del modelo (mayor = más probable positivo).

Métrica principal: TPR (recall) sobre positivos reservados en un punto de operación fijo.
Control: P(ŷ=1 | U). Lee y Liu (2003) solo sirve para ORDENAR modelos, porque supone
SCAR y aquí hay sesgo químico en lo que se ha sintetizado.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def _as_binary(arr: Iterable[int], name: str) -> np.ndarray:
    """Convierte a array entero y verifica que solo contenga 0 y 1.

    La validación se hace ANTES de convertir: si no, una probabilidad como 0.3 se
    truncaría en silencio a 0 y pasaría por una predicción binaria.
    """
    raw = np.asarray(arr)
    if not np.isin(raw, (0, 1)).all():
        raise ValueError(f"`{name}` debe contener solo 0 y 1 (¿se pasaron puntuaciones?).")
    return raw.astype(int)


def true_positive_rate(s: Iterable[int], y_pred: Iterable[int]) -> float:
    """TPR = fracción de positivos etiquetados predichos como positivos.

    Devuelve NaN si no hay positivos (p. ej., un fold leave-one-group-out sin positivos).
    """
    s, y_pred = _as_binary(s, "s"), _as_binary(y_pred, "y_pred")
    pos = s == 1
    return float(y_pred[pos].mean()) if pos.any() else math.nan


def predicted_positive_rate(y_pred: Iterable[int]) -> float:
    """P(ŷ = 1) sobre todas las composiciones evaluadas (positivas + no etiquetadas)."""
    y_pred = _as_binary(y_pred, "y_pred")
    return float(y_pred.mean()) if y_pred.size else math.nan


def unlabeled_positive_rate(s: Iterable[int], y_pred: Iterable[int]) -> float:
    """P(ŷ = 1 | U): fracción de no etiquetados predichos como positivos (métrica de control)."""
    s, y_pred = _as_binary(s, "s"), _as_binary(y_pred, "y_pred")
    unl = s == 0
    return float(y_pred[unl].mean()) if unl.any() else math.nan


def lee_liu_score(s: Iterable[int], y_pred: Iterable[int]) -> float:
    """Métrica de Lee y Liu (2003): recall² / P(ŷ = 1).

    Supone SCAR, que aquí NO se cumple: úsese solo para ordenar modelos, nunca como
    estimación absoluta. Devuelve NaN si P(ŷ = 1) = 0 (forma 0/0).
    """
    recall = true_positive_rate(s, y_pred)
    ppr = predicted_positive_rate(y_pred)
    if math.isnan(recall) or not ppr:
        return math.nan
    return recall**2 / ppr


def threshold_for_ppr(scores: Iterable[float], target_ppr: float) -> float:
    """Umbral tal que la fracción de puntuaciones >= umbral se acerque a ``target_ppr``.

    Se predicen positivos los ``k = ceil(target_ppr · n)`` puntuaciones más altas. Con
    empates en el umbral, la fracción real puede superar el objetivo; por eso
    :func:`tpr_at_ppr` devuelve también la P(ŷ = 1) alcanzada.
    """
    if not 0.0 <= target_ppr <= 1.0:
        raise ValueError("`target_ppr` debe estar en [0, 1].")
    scores = np.sort(np.asarray(scores, dtype=float))[::-1]  # orden descendente
    k = math.ceil(round(target_ppr * scores.size, 9))  # round evita errores de coma flotante
    if k == 0:
        return math.inf  # nada se predice positivo
    return float(scores[k - 1])


def threshold_for_recall(
    s: Iterable[int], scores: Iterable[float], target_recall: float = 0.90
) -> float:
    """Umbral que deja al menos ``target_recall`` de los positivos por encima.

    Es el punto de operación principal preregistrado (sección 7.1 de docs/preregistro.md):
    en el bucle interno se calcula sobre las puntuaciones FUERA DE FOLD de los positivos y
    luego se aplica sin cambios al test externo. Equivale a :func:`threshold_for_ppr`
    restringido a los positivos: con n positivos se toman los ``ceil(target_recall · n)``
    de mayor puntuación (para 0,90, el percentil 10 de sus puntuaciones).

    Raises:
        ValueError: si no hay positivos o ``target_recall`` no está en (0, 1].
    """
    if not 0.0 < target_recall <= 1.0:
        raise ValueError("`target_recall` debe estar en (0, 1].")
    s = _as_binary(s, "s")
    scores = np.asarray(scores, dtype=float)
    if not (s == 1).any():
        raise ValueError("No hay positivos para calibrar el umbral.")
    return threshold_for_ppr(scores[s == 1], target_recall)


def tpr_at_ppr(s: Iterable[int], scores: Iterable[float], target_ppr: float) -> tuple[float, float]:
    """TPR a una P(ŷ = 1) fija, para comparar modelos con umbrales distintos (incluido τ).

    Returns:
        (tpr, ppr_alcanzada)
    """
    scores = np.asarray(scores, dtype=float)
    thr = threshold_for_ppr(scores, target_ppr)
    y_pred = (scores >= thr).astype(int)
    return true_positive_rate(s, y_pred), predicted_positive_rate(y_pred)


def expected_hits_at_ppr(s: Iterable[int], scores: Iterable[float], target_ppr: float) -> float:
    """Aciertos esperados (positivos predichos) a una P(ŷ = 1) EXACTA, repartiendo empates.

    Se predicen positivos las ``target_ppr · n`` puntuaciones más altas. Si el corte cae dentro
    de un bloque de puntuaciones empatadas, ese bloque entra en la fracción justa y aporta la
    misma fracción de sus positivos: es el valor esperado al romper los empates al azar
    (interpolación lineal, como en las curvas ROC). Sin empates en el corte coincide con
    :func:`tpr_at_ppr`. Con puntuaciones binarias (p. ej. el balance de carga) evita que todo
    el bloque empatado entre de golpe y la P(ŷ = 1) real supere el objetivo.

    Devuelve un recuento (no una tasa) para micro-agregar sobre folds: Σ aciertos / Σ positivos.
    """
    if not 0.0 <= target_ppr <= 1.0:
        raise ValueError("`target_ppr` debe estar en [0, 1].")
    s = _as_binary(s, "s")
    scores = np.asarray(scores, dtype=float)
    k = target_ppr * scores.size  # número (fraccionario) de predicciones positivas
    taken = hits = 0.0
    for v in np.unique(scores)[::-1]:  # bloques de puntuación, de mayor a menor
        block = scores == v
        n_block, pos_block = int(block.sum()), int((s[block] == 1).sum())
        if taken + n_block <= k + 1e-12:
            taken, hits = taken + n_block, hits + pos_block
        else:
            hits += pos_block * (k - taken) / n_block
            break
    return float(hits)


def hit_probability_at_ppr(scores: Iterable[float], target_ppr: float) -> np.ndarray:
    """Probabilidad de que cada fila se prediga positiva a una P(ŷ = 1) EXACTA.

    Versión por fila de :func:`expected_hits_at_ppr`, con el mismo reparto de empates: 1 para
    los bloques de puntuación que entran enteros, la fracción justa para el bloque donde cae el
    corte y 0 para el resto. Sumada sobre los positivos da los aciertos esperados. Sirve para
    las diferencias pareadas por composición a igual P(ŷ = 1) (preregistro §7.2 y §8).
    """
    if not 0.0 <= target_ppr <= 1.0:
        raise ValueError("`target_ppr` debe estar en [0, 1].")
    scores = np.asarray(scores, dtype=float)
    prob = np.zeros(scores.size)
    k = target_ppr * scores.size  # número (fraccionario) de predicciones positivas
    taken = 0.0
    for v in np.unique(scores)[::-1]:  # bloques de puntuación, de mayor a menor
        block = scores == v
        n_block = int(block.sum())
        if taken + n_block <= k + 1e-12:
            prob[block] = 1.0
            taken += n_block
        else:
            prob[block] = (k - taken) / n_block  # fracción del bloque empatado que entra
            break
    return prob


def tpr_ppr_curve(
    s: Iterable[int], scores: Iterable[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Curva TPR frente a P(ŷ = 1), barriendo todos los umbrales distintos.

    Returns:
        (umbrales en orden descendente, tpr, ppr). Empieza en (inf, 0, 0).
    """
    s = _as_binary(s, "s")
    scores = np.asarray(scores, dtype=float)
    thresholds = np.concatenate(([np.inf], np.unique(scores)[::-1]))
    pred = scores[None, :] >= thresholds[:, None]  # una fila de predicciones por umbral
    n_pos = max(int(s.sum()), 1)
    tpr = (pred & (s == 1)).sum(axis=1) / n_pos
    ppr = pred.mean(axis=1)
    return thresholds, tpr, ppr


def micro_tpr(per_fold: Iterable[tuple[Iterable[int], Iterable[int]]]) -> float:
    """TPR micro-agregado sobre folds: Σ verdaderos positivos / Σ positivos.

    Se usa en ``loeo_A`` y ``loeo_B``, donde muchos folds tienen pocos positivos y el
    promedio de TPR por fold estaría dominado por folds pequeños.
    """
    tp = n_pos = 0
    for s, y_pred in per_fold:
        s, y_pred = _as_binary(s, "s"), _as_binary(y_pred, "y_pred")
        tp += int(((s == 1) & (y_pred == 1)).sum())
        n_pos += int((s == 1).sum())
    return tp / n_pos if n_pos else math.nan


def lower_bound_metrics(
    s: Iterable[int], y_pred: Iterable[int], scores: Iterable[float] | None = None
) -> dict[str, float]:
    """Accuracy, F1 y ROC-AUC tratando U como negativo.

    Son COTAS INFERIORES (parte de U son positivos aún no sintetizados). Las claves
    llevan el sufijo ``_lower_bound`` para que no puedan reportarse por error como
    métricas absolutas.
    """
    s, y_pred = _as_binary(s, "s"), _as_binary(y_pred, "y_pred")
    out = {
        "accuracy_lower_bound": float(accuracy_score(s, y_pred)),
        "f1_lower_bound": float(f1_score(s, y_pred, zero_division=0)),
    }
    if scores is not None and len(np.unique(s)) == 2:
        out["roc_auc_lower_bound"] = float(roc_auc_score(s, np.asarray(scores, dtype=float)))
    return out
