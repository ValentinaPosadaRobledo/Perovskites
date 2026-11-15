"""F3.2: estimación de la prior de clase a partir de datos PU (Elkan–Noto y TIcE).

Notación (Bekker y Davis, 2020, revisión de PU learning):
    s     1 = positiva etiquetada (L), 0 = no etiquetada (U). Las U NO son negativas.
    c     frecuencia de etiquetado, P(s = 1 | y = 1).
    α     prior de clase sobre todo el conjunto, P(y = 1).
    π_U   fracción de positivas ocultas entre las no etiquetadas.

Bajo SCAR (las positivas se etiquetan al azar, con probabilidad c constante):

    α = P(s = 1) / c = n_L / (c · N)        π_U = n_L · (1 − c) / (c · n_U)

**Aquí SCAR no se cumple:** las perovskitas conocidas se concentran en las químicas más
estudiadas. Por eso π solo se usa de forma descriptiva y en análisis de sensibilidad
(decisión de la autora, 2026-11-14); nada del protocolo principal depende de ella.

Métodos (preregistro §7, "≥ 2 métodos"):

1. **Elkan–Noto** (Elkan y Noto, 2008): un clasificador calibrado g(x) ≈ P(s = 1 | x),
   entrenado para separar L de U, cumple g(x) = c en las positivas. c se estima como la media
   de g sobre positivas etiquetadas que el clasificador NO vio (estimador e1). Aquí se hace
   con validación cruzada: cada positiva se puntúa con el modelo de los otros folds.
2. **TIcE** (Bekker y Davis, 2018): c es la misma en cualquier subregión del espacio, y la
   fracción de etiquetadas de una subregión es una cota inferior de c (iguala a c donde solo
   hay positivas). Un árbol inducido de forma voraz busca la subregión con la mayor cota.
   Se sigue Algorithm 1 del artículo con sus valores por defecto (:func:`tice_label_frequency`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedKFold


# ---------------------------------------------------------------------------
# De la frecuencia de etiquetado a la prior
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PriorEstimate:
    """Resultado de una estimación: c, α, π_U y si hubo que recortar a [0, 1]."""

    c: float
    alpha: float
    pi_u: float
    n_labeled: int
    n_unlabeled: int
    clipped: bool

    @property
    def hidden_positives(self) -> float:
        """N.º esperado de positivas ocultas entre las no etiquetadas (π_U · n_U)."""
        return self.pi_u * self.n_unlabeled


def prior_from_label_frequency(c: float, n_labeled: int, n_unlabeled: int) -> PriorEstimate:
    """α y π_U a partir de c (relaciones de SCAR, ver docstring del módulo).

    c no puede ser menor que n_L / N (implicaría α > 1) ni mayor que 1. Si lo es, se recorta
    a ese intervalo y se marca `clipped = True` para que el informe lo muestre.
    """
    if n_labeled <= 0 or n_unlabeled < 0:
        raise ValueError("Hacen falta positivas etiquetadas (n_labeled > 0).")
    n = n_labeled + n_unlabeled
    c_min = n_labeled / n  # con c = n_L / N, todas las U serían positivas (α = 1)
    clipped = not (c_min <= c <= 1.0)
    c_eff = min(max(c, c_min), 1.0)
    alpha = n_labeled / (c_eff * n)
    pi_u = n_labeled * (1.0 - c_eff) / (c_eff * n_unlabeled) if n_unlabeled else 0.0
    return PriorEstimate(c=float(c), alpha=float(alpha), pi_u=float(pi_u),
                         n_labeled=int(n_labeled), n_unlabeled=int(n_unlabeled),
                         clipped=bool(clipped))  # fmt: skip


# ---------------------------------------------------------------------------
# Elkan–Noto (2008), estimador e1 con validación cruzada
# ---------------------------------------------------------------------------
def elkan_noto_label_frequency(
    X: np.ndarray,
    s: np.ndarray,
    estimator: BaseEstimator,
    n_splits: int = 5,
    random_state: int | None = None,
) -> tuple[float, np.ndarray]:
    """c = media de g(x) sobre las positivas etiquetadas, cada una puntuada fuera de fold.

    `estimator` debe dar probabilidades CALIBRADAS de P(s = 1 | x) (p. ej. un
    `CalibratedClassifierCV` dentro de un `Pipeline`): Elkan–Noto lo exige, porque usa el
    valor de la probabilidad y no solo el orden. Se entrena con la proporción natural L/U
    (sin equilibrar), que es la que define P(s = 1 | x).

    Devuelve (c, puntuaciones fuera de fold de todas las filas), para poder revisar la
    calibración y la distribución de g en L y en U.
    """
    X, s = np.asarray(X), np.asarray(s).astype(int)
    if set(np.unique(s)) - {0, 1} or s.sum() == 0:
        raise ValueError("`s` debe tener 0 y 1, con al menos una positiva etiquetada.")
    oof = np.full(s.size, np.nan)
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rng = np.random.default_rng(random_state)
    for train, test in folds.split(X, s):
        model = clone(estimator)
        # Semilla del modelo: si el estimador la admite, se fija desde `random_state`.
        if random_state is not None and "random_state" in model.get_params():
            model.set_params(random_state=random_state)
        # Se barajan las filas de entrenamiento: una CV interna sin barajar (como la de
        # `CalibratedClassifierCV`) heredaría el orden del archivo y calibraría con folds no
        # representativos (p. ej. si las positivas van primero).
        train = rng.permutation(train)
        model.fit(X[train], s[train])
        oof[test] = model.predict_proba(X[test])[:, 1]
    c = float(np.mean(oof[s == 1]))  # e1: media sobre las positivas no vistas
    return c, oof


# ---------------------------------------------------------------------------
# TIcE (Bekker y Davis, 2018), Algorithm 1
# ---------------------------------------------------------------------------
def tice_delta(n_estimation: int) -> float:
    """δ de la Ec. (8) de Bekker y Davis (2018): max(0,025; 1 / (1 + 0,004 · T)).

    Equivale a exigir T_R = min(1000; 0,1 · T) ejemplos para una cota con error ε = 0,1.
    Se calcula una vez por fold, con T = n.º de filas de los datos de estimación.
    """
    return max(0.025, 1.0 / (1.0 + 0.004 * n_estimation))


def tice_lower_bound(n_labeled: int, n_total: int, delta: float, c_prior: float = 0.5) -> float:
    """Cota inferior de c en una subregión (Ec. 6–7 de Bekker y Davis, 2018).

    El n.º de etiquetadas L de una región con T filas sigue una Binomial(P, c) con P ≤ T
    positivas. Por la desigualdad de Chebyshev unilateral (Cantelli), con probabilidad ≥ 1 − δ:

        c ≥ L / T − √(ĉ (1 − ĉ) (1 − δ) / (δ T))

    con ĉ una estimación previa de c (con ĉ = 0,5 es el peor caso, ½ · √((1 − δ)/(δ T))).
    """
    if n_total <= 0:
        return -math.inf
    return n_labeled / n_total - math.sqrt(
        c_prior * (1.0 - c_prior) * (1.0 - delta) / (delta * n_total)
    )


def tice_discretize(X: np.ndarray, n_bins: int = 4, binning: str = "uniform") -> np.ndarray:
    """Discretiza cada feature numérica en `n_bins` intervalos (valores 0 … n_bins − 1).

    - "uniform" (como el artículo): se escala a [0, 1] y el rango se parte en `n_bins`
      partes iguales. Es sensible a valores extremos (casi todo puede caer en un intervalo).
    - "quantile" (sensibilidad, en la línea de dos Reis et al., 2021): cortes por cuantiles,
      con intervalos de tamaño parecido.

    Las features constantes quedan en un solo intervalo (no se pueden usar para partir).
    """
    X = np.asarray(X, dtype=float)
    if np.isnan(X).any():
        raise ValueError("TIcE no admite huecos: imputar antes.")
    out = np.zeros(X.shape, dtype=np.int8)
    for j in range(X.shape[1]):
        col = X[:, j]
        lo, hi = col.min(), col.max()
        if hi <= lo:
            continue  # constante
        if binning == "uniform":
            scaled = (col - lo) / (hi - lo)
            out[:, j] = np.minimum((scaled * n_bins).astype(int), n_bins - 1)  # 1,0 → último
        elif binning == "quantile":
            edges = np.unique(np.quantile(col, np.linspace(0, 1, n_bins + 1)[1:-1]))
            out[:, j] = np.searchsorted(edges, col, side="right")
        else:
            raise ValueError(f"binning desconocido: {binning!r} (uniform o quantile)")
    return out


def _tice_fold(
    B_tree: np.ndarray,
    s_tree: np.ndarray,
    B_est: np.ndarray,
    s_est: np.ndarray,
    c_prior: float,
    k: float,
    max_splits: int,
) -> float:
    """Líneas 5–16 de Algorithm 1 para un par (datos del árbol, datos de estimación).

    Los nodos se guardan como máscaras booleanas sobre los dos conjuntos y el conjunto de
    atributos aún disponibles (una feature discretizada ya usada es constante en el subárbol).
    """
    delta = tice_delta(s_est.size)  # línea 5: δ con T de los datos de estimación
    c_best = s_est.sum() / s_est.size  # línea 6: L / T sin corrección
    n_feat = B_tree.shape[1]
    root = (np.ones(s_tree.size, bool), np.ones(s_est.size, bool), frozenset(range(n_feat)))
    queue = [root]  # línea 7
    j = 0
    while j < max_splits and queue:  # línea 8
        # Línea 9: el nodo más prometedor según su cota inferior EN LOS DATOS DEL ÁRBOL.
        bounds = [tice_lower_bound(int(s_tree[mt].sum()), int(mt.sum()), delta, c_prior)
                  for mt, _, _ in queue]  # fmt: skip
        m_tree, m_est, atts = queue.pop(int(np.argmax(bounds)))  # líneas 9–10
        j += 1
        # Línea 11: atributo con el máximo bepp_k = L / (T + k) entre sus valores (árbol).
        best_att, best_score = None, -math.inf
        st = s_tree[m_tree]
        for a in sorted(atts):
            col = B_tree[m_tree, a]
            values = np.unique(col)
            if values.size < 2:
                continue  # no divide la región
            score = max(st[col == v].sum() / ((col == v).sum() + k) for v in values)
            if score > best_score:
                best_att, best_score = a, score
        if best_att is None:
            continue  # nodo indivisible: cuenta como split consumido, igual que en el bucle
        rest = atts - {best_att}
        # Líneas 12–15 recorren todo el dominio del atributo: también los valores que solo
        # aparecen en los datos de estimación (su cota cuenta aunque no se puedan partir).
        domain = np.union1d(B_tree[m_tree, best_att], B_est[m_est, best_att])
        for v in domain:
            child_tree = m_tree & (B_tree[:, best_att] == v)
            child_est = m_est & (B_est[:, best_att] == v)
            if child_tree.any():  # un hijo sin datos del árbol no se puede puntuar ni partir
                queue.append((child_tree, child_est, rest))  # línea 13
            n_e = int(child_est.sum())
            if n_e:  # línea 14: cota en los datos de estimación del hijo
                c_low = tice_lower_bound(int(s_est[child_est].sum()), n_e, delta, c_prior)
                c_best = max(c_best, c_low)  # línea 15
    return float(c_best)


def tice_label_frequency(
    X: np.ndarray,
    s: np.ndarray,
    n_folds: int = 5,
    k: float = 5.0,
    max_splits: int = 500,
    n_bins: int = 4,
    binning: str = "uniform",
    random_state: int | None = None,
) -> tuple[float, dict]:
    """TIcE (Bekker y Davis, 2018, Algorithm 1) con sus valores por defecto.

    Valores del artículo: k = 5 (max-bepp), M = 500 particiones, 5 folds, δ por la Ec. (8)
    en cada fold y features numéricas escaladas a [0, 1] con el rango partido en 4.

    1. Los datos se dividen en `n_folds` subconjuntos aleatorios de igual tamaño; en cada
       fold, UNO es el de inducción del árbol y el resto, el de estimación.
    2. Primera pasada con ĉ = 0,5 (peor caso de la cota); la media de los folds es c_prior.
    3. Segunda pasada con los mismos folds y ĉ = c_prior (cota más ajustada). Su media es ĉ.

    `X` no debe tener huecos (se imputan antes). Devuelve (ĉ, detalle con las estimaciones
    por fold de las dos pasadas).

    Propiedad conocida (comprobada con datos sintéticos SCAR de N = 2000): la línea 14 calcula
    la cota en todos los hijos, también en los muy pequeños, y un nodo pequeño y totalmente
    etiquetado ya da una cota de ≈ 0,3. Con c real baja (≈ 0,1) TIcE sobrestima c; con c ≥ 0,5
    la estimación queda a menos de 0,08. En `train_pool`, c ≥ n_L / N ≈ 0,21 por construcción.
    """
    from sklearn.model_selection import KFold

    s = np.asarray(s).astype(int)
    if set(np.unique(s)) - {0, 1} or s.sum() == 0:
        raise ValueError("`s` debe tener 0 y 1, con al menos una positiva etiquetada.")
    B = tice_discretize(X, n_bins=n_bins, binning=binning)
    # "k random equal-sized subsets": KFold barajado (sin estratificar, como el artículo).
    folds = list(KFold(n_splits=n_folds, shuffle=True, random_state=random_state).split(B))
    c_hat, passes = 0.5, []  # línea 1
    for _ in range(2):  # línea 2: dos pasadas
        per_fold = [
            _tice_fold(B[tree], s[tree], B[est], s[est], c_hat, k, max_splits)
            for est, tree in folds
        ]  # el subconjunto "de test" de KFold es el del árbol
        passes.append(per_fold)
        c_hat = float(np.mean(per_fold))  # línea 17
    return c_hat, {"first_pass": passes[0], "second_pass": passes[1]}


# ---------------------------------------------------------------------------
# Uso de π en la sensibilidad: precisión estimada en el punto de operación
# ---------------------------------------------------------------------------
def precision_under_scar(tpr: float, ppr_u: float, pi_u: float, n_pos: int, n_u: int) -> dict:
    """Precisión esperada si las positivas ocultas se detectan con la misma TPR (SCAR).

    En el test hay n_pos positivas etiquetadas y n_u no etiquetadas, de las que π_U · n_u son
    positivas ocultas. Con TPR t y P(ŷ = 1 | U) = q:

    - positivas ocultas predichas ≈ t · π_U · n_u (si se detectan como las etiquetadas);
    - precisión entre las U predichas = min(1, t · π_U / q);
    - precisión global = (t · n_pos + t · π_U · n_u) / (t · n_pos + q · n_u).

    Es una aproximación de sensibilidad, no una métrica del protocolo: SCAR no se cumple.
    """
    hidden_hits = tpr * pi_u * n_u
    predicted_u = ppr_u * n_u
    precision_u = min(1.0, hidden_hits / predicted_u) if predicted_u else math.nan
    denom = tpr * n_pos + predicted_u
    precision_all = min(1.0, (tpr * n_pos + hidden_hits) / denom) if denom else math.nan
    return {"precision_u_scar": precision_u, "precision_scar": precision_all}
