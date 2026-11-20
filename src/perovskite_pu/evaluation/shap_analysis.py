"""F4.3: SHAP del modelo PU `pre_dft` final y su estabilidad entre semillas.

Decisiones de la autora (2026-11-20; `configs/experiments/f4_3_shap.yaml`):

- **Modelo:** el PU `pre_dft` entrenado con todo `train_pool` en cada semilla, con la misma
  búsqueda interna y selección que F3.3 (variante `final` de F3.4). Se reconstruye el candidato
  elegido con sus mismas semillas y se comprueba que puntúa igual que `select_and_fit_pu`.
- **SHAP:** `TreeExplainer` *path-dependent* (el exacto de Lundberg et al., 2020) sobre la
  salida cruda de cada bolsa: probabilidad con Random Forest y log-odds con Gradient Boosting.
  La media de los SHAP de las bolsas explica la media de sus salidas crudas, que con RF es la
  puntuación del modelo y con GB la media de log-odds (se informa su Spearman con la
  puntuación). Se comprueba la aditividad: base + Σ SHAP = salida media. El modo
  intervencional en escala de probabilidad se descartó porque con GB no es aditivo (desvío de
  hasta 0,016 en la puntuación; 2026-11-20). Como las unidades dependen del aprendiz, la
  importancia también se da relativa (fracción del total de cada semilla).
- **Grupos:** el SHAP de un grupo es la suma de los SHAP con signo de sus features (también
  aditivo); su importancia, la media de su valor absoluto.
- **Estabilidad:** ranking por importancia global en cada semilla; Spearman entre los pares de
  semillas, Jaccard del top-k y frecuencia de cada feature o grupo en el top-k.

Descriptivo y fuera del núcleo mínimo. Solo lee `train_pool`.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def feature_groups(names: Sequence[str], groups_cfg: Mapping[str, Any]) -> list[str]:
    """Grupo de cada feature: patrones con nombre en orden y, si no, la propiedad de MAGPIE.

    Raises:
        ValueError: si alguna feature no cae en ningún grupo.
    """
    magpie = re.compile(groups_cfg["magpie"])
    named = [(re.compile(r["pattern"]), r["group"]) for r in groups_cfg.get("named", [])]
    out = []
    for n in names:
        g = next((grp for pat, grp in named if pat.search(n)), None)
        if g is None and (m := magpie.match(n)):
            g = m.group("prop")
        if g is None:
            raise ValueError(f"Feature sin grupo: {n!r}")
        out.append(g)
    return out


def _class1(values: Any, expected: Any) -> tuple[np.ndarray, float]:
    """SHAP y valor base de la clase 1, sea cual sea el formato que devuelve `shap`."""
    v = np.asarray(values)
    if isinstance(values, list):  # formato antiguo: una matriz por clase
        v = np.asarray(values[1])
    elif v.ndim == 3:  # (n, features, clases)
        v = v[..., 1]
    e = np.ravel(np.asarray(expected, dtype=float))
    return v, float(e[1] if e.size == 2 else e[0])


def raw_output(model: Any, X: np.ndarray) -> tuple[np.ndarray, str]:
    """Salida cruda que explica TreeSHAP: probabilidad (RF) o log-odds (GB)."""
    if hasattr(model, "decision_function") and not hasattr(model, "estimators_samples_"):
        return np.asarray(model.decision_function(X), dtype=float), "log_odds"
    return np.asarray(model.predict_proba(X))[:, 1], "probability"


def bagging_shap(
    pipeline: Any, X: np.ndarray, feature_names: Sequence[str]
) -> tuple[pd.DataFrame, float, np.ndarray, np.ndarray, str]:
    """SHAP del pipeline PU (preprocesado + bagging), media de las bolsas.

    Se explica la salida cruda de cada bolsa en el espacio ya preprocesado (tras la imputación
    con indicador y el filtro de varianza, ajustados en el mismo train). Devuelve (SHAP por
    composición y feature transformada, valor base, salida cruda media de las bolsas,
    puntuación del modelo, espacio de salida).
    """
    import shap

    pre = pipeline[:-1]
    Xt = pre.transform(X)
    names = list(pre.get_feature_names_out(list(feature_names)))
    total, base, out = np.zeros_like(Xt, dtype=float), 0.0, np.zeros(len(Xt))
    bags = pipeline[-1].estimators_
    spaces = set()
    for model in bags:
        ex = shap.TreeExplainer(model)  # path-dependent: exacto sobre la salida cruda
        v, e = _class1(ex.shap_values(Xt, check_additivity=False), ex.expected_value)
        r, space = raw_output(model, Xt)
        total, base, out = total + v, base + e, out + r
        spaces.add(space)
    if len(spaces) != 1:
        raise ValueError(f"Bolsas con salidas de espacios distintos: {spaces}")
    n = len(bags)
    score = np.asarray(pipeline.predict_proba(X))[:, 1]
    return pd.DataFrame(total / n, columns=names), base / n, out / n, score, spaces.pop()


def importance_tables(
    shap_values: pd.DataFrame, groups: Sequence[str], labels: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Importancia por feature y por grupo: media de |SHAP| en todas, en P y en U.

    También la media con signo en P y en U (dirección del efecto). El SHAP de un grupo es la
    suma de los de sus features, antes del valor absoluto.
    """
    pos = np.asarray(labels) == 1

    def summarize(v: pd.DataFrame) -> pd.DataFrame:
        a = v.abs()
        return pd.DataFrame({
            "mean_abs_all": a.mean(), "mean_abs_P": a[pos].mean(), "mean_abs_U": a[~pos].mean(),
            "mean_signed_P": v[pos].mean(), "mean_signed_U": v[~pos].mean(),
        })  # fmt: skip

    feat = summarize(shap_values).rename_axis("feature").reset_index()
    feat["share_all"] = feat["mean_abs_all"] / feat["mean_abs_all"].sum()  # importancia relativa
    feat.insert(1, "group", list(groups))
    grouped = shap_values.T.groupby(list(groups)).sum().T  # suma con signo por grupo
    grp = summarize(grouped).rename_axis("group").reset_index()
    grp["share_all"] = grp["mean_abs_all"] / grp["mean_abs_all"].sum()
    grp.insert(1, "n_features", grp["group"].map(pd.Series(groups).value_counts()))
    return feat, grp


def stability(
    importance_by_seed: pd.DataFrame, item: str, rank_by: str, top_k: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estabilidad del ranking entre semillas.

    Devuelve (una fila por par de semillas con Spearman y Jaccard del top-k; una fila por
    elemento con su importancia media, DE, rango medio y frecuencia en el top-k).
    """
    wide = importance_by_seed.pivot(index=item, columns="seed", values=rank_by)
    ranks = wide.rank(ascending=False, method="average")
    top = {s: set(wide[s].nlargest(top_k).index) for s in wide.columns}
    pairs = []
    for a, b in itertools.combinations(wide.columns, 2):
        rho = stats.spearmanr(wide[a], wide[b]).statistic
        jac = len(top[a] & top[b]) / len(top[a] | top[b])
        pairs.append({"seed_a": a, "seed_b": b, "spearman": float(rho),
                      f"jaccard_top{top_k}": jac})  # fmt: skip
    per_item = pd.DataFrame({
        f"{rank_by}_mean": wide.mean(axis=1), f"{rank_by}_sd": wide.std(axis=1, ddof=1),
        "rank_mean": ranks.mean(axis=1), "rank_min": ranks.min(axis=1),
        "rank_max": ranks.max(axis=1),
        f"freq_top{top_k}": pd.Series({i: sum(i in t for t in top.values()) / len(top)
                                       for i in wide.index}),
    }).sort_values(f"{rank_by}_mean", ascending=False)  # fmt: skip
    return pd.DataFrame(pairs), per_item.rename_axis(item).reset_index()
