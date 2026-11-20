"""F4.2: pruebas estadísticas preregistradas (docs/preregistro.md, secciones 8 y 9).

- **Unidad de análisis:** diferencias pareadas por composición. Para cada positivo c,
  h_M,e(c) es la fracción de las semillas en que el modelo M acierta con c en el esquema e
  (cada positivo se evalúa exactamente una vez por semilla y esquema). Así,
  TPR_M,e = media_c h_M,e(c), que coincide con la TPR micro-agregada.
- **H1, H2, H3** (esquemas confirmatorios `loeo_A`, `loeo_B`, `family`, `cluster`):
  bootstrap jerárquico pareado. Nivel 1, unidades químicas con reemplazo (elemento A en
  `loeo_A`; elemento B en los demás; en `family`, estratificado por familia). Nivel 2,
  positivos con reemplazo dentro de cada unidad elegida. B = 10 000, IC por percentiles.
- **Comparaciones secundarias** en `random` y `cluster`: t corregido de Nadeau y Bengio.
- **Multiplicidad:** Holm dentro de cada hipótesis, sobre sus 4 esquemas; conclusión global
  por intersección-unión.

Este módulo no lee archivos: recibe tablas ya cargadas, para poder usarlo igual con los
folds de validación cruzada y, en F4.1, con el conjunto de prueba final.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps

from perovskite_pu.evaluation.metrics import hit_probability_at_ppr

CONFIRMATORY_SCHEMES = ("loeo_A", "loeo_B", "family", "cluster")
# Hipótesis e índices fijos para derivar el flujo aleatorio de cada bootstrap (ver
# `bootstrap_rng`): cambiar el orden cambiaría los números, así que no se reordenan.
HYPOTHESES = ("H1", "H2", "H2_sensitivity", "H3", "H3_symmetry")
BOOTSTRAP_SCHEMES = ("random", "group_system", "loeo_A", "loeo_B", "family", "cluster", "temporal")


# ---------------------------------------------------------------------------
# Aciertos por composición
# ---------------------------------------------------------------------------


def composition_hits(raw: pd.DataFrame, model: str, scheme: str) -> pd.Series:
    """h(c): fracción de semillas en que el modelo acierta con cada positivo, en un esquema.

    `raw` tiene una fila por (modelo, esquema, semilla, fold, composición), con las columnas
    `baseline`, `scheme`, `seed`, `composition`, `label` y `pred` (formato de `raw/seed_n.csv`).

    Raises:
        ValueError: si un positivo no aparece exactamente una vez en cada semilla.
    """
    d = raw[(raw["baseline"] == model) & (raw["scheme"] == scheme) & (raw["label"] == 1)]
    if d.empty:
        raise ValueError(f"Sin positivos para {model} en {scheme}.")
    _check_once_per_seed(d, f"{model}/{scheme}")
    return d.groupby("composition")["pred"].mean().rename(model)


def matched_ppr_hits(
    raw_model: pd.DataFrame,
    raw_ref: pd.DataFrame,
    scheme: str,
    exclude: Iterable[str] = (),
) -> pd.DataFrame:
    """h(c) del modelo umbralizado a la P(ŷ = 1) de la referencia, fold a fold (§7.2).

    En cada semilla y fold, sobre las composiciones donde la referencia está definida (menos
    `exclude`), se toma la P(ŷ = 1) de la referencia y el modelo se umbraliza para igualarla,
    repartiendo empates (:func:`hit_probability_at_ppr`). Es el contraste de H2 (modelo PU
    frente a τ) y de H3 (grafo frente al PU composicional).

    `raw_model` y `raw_ref` tienen el formato de `raw/seed_n.csv` y una sola línea base cada uno.
    Devuelve una fila por positivo con `h_model` y `h_ref` (media sobre semillas).
    """
    excl = set(exclude)
    m = raw_model[raw_model["scheme"] == scheme]
    r = raw_ref[(raw_ref["scheme"] == scheme) & ~raw_ref["composition"].isin(excl)]
    rows = []
    for (seed, fold), rf in r.groupby(["seed", "fold"], sort=True):
        mf = m[(m["seed"] == seed) & (m["fold"] == fold) & m["composition"].isin(rf["composition"])]
        # Mismas composiciones en los dos modelos: los folds congelados son idénticos.
        if len(mf) != len(rf) or set(mf["composition"]) != set(rf["composition"]):
            raise ValueError(f"Folds distintos en {scheme}, semilla {seed}, fold {fold}.")
        prob = hit_probability_at_ppr(mf["score"].to_numpy(float), float(rf["pred"].mean()))
        pos_m = mf["label"].to_numpy(int) == 1
        hit_m = pd.Series(prob[pos_m], index=mf["composition"].to_numpy()[pos_m])
        pos_r = rf[rf["label"] == 1].set_index("composition")["pred"].astype(float)
        both = pd.DataFrame({"h_model": hit_m, "h_ref": pos_r})
        if both.isna().any().any():
            raise ValueError(f"Etiquetas distintas en {scheme}, semilla {seed}, fold {fold}.")
        rows.append(both.assign(seed=seed))
    if not rows:
        raise ValueError(f"Sin composiciones comparables en {scheme}.")
    long = pd.concat(rows).rename_axis("composition").reset_index()
    _check_once_per_seed(long, f"{scheme} (igual P(ŷ=1))")
    return long.groupby("composition")[["h_model", "h_ref"]].mean()


def fold_tpr_at_ppr_diffs(
    raw_a: pd.DataFrame, raw_b: pd.DataFrame, scheme: str, ppr: float
) -> pd.DataFrame:
    """TPR de dos modelos a la misma P(ŷ = 1) fija, fold a fold (Nadeau–Bengio, §9).

    En cada semilla y fold se toman las composiciones puntuadas por **los dos** modelos (el
    subconjunto común: p. ej. las de τ definido o las de `post_dft`), se umbraliza cada modelo a
    la P(ŷ = 1) `ppr` sobre ese subconjunto, repartiendo empates (:func:`hit_probability_at_ppr`),
    y se calcula la TPR. Los folds sin positivos en el subconjunto común se omiten.

    `raw_a` y `raw_b` tienen el formato de `raw/seed_n.csv` y una sola línea base cada uno.
    Devuelve una fila por (semilla, fold) con `n_test`, `n_pos`, `tpr_a`, `tpr_b` y
    `diff` = `tpr_a` − `tpr_b`; `n_common` es el tamaño del subconjunto común en la semilla
    (para el cociente n_test / n_train de la corrección).

    Raises:
        ValueError: si una composición cambia de fold o de etiqueta entre los dos modelos.
    """
    a = raw_a[raw_a["scheme"] == scheme]
    b = raw_b[raw_b["scheme"] == scheme]
    keys = ["seed", "fold", "composition"]
    both = a[[*keys, "label", "score"]].merge(
        b[[*keys, "label", "score"]], on=["seed", "composition"], suffixes=("_a", "_b")
    )
    # Los folds congelados son los mismos para todos los modelos: una composición común debe
    # caer en el mismo fold y con la misma etiqueta en los dos.
    if (both["fold_a"] != both["fold_b"]).any() or (both["label_a"] != both["label_b"]).any():
        raise ValueError(f"Folds o etiquetas distintos entre los dos modelos en {scheme}.")
    both = both.rename(columns={"fold_a": "fold", "label_a": "label"})
    n_common = both.groupby("seed").size()
    rows = []
    for (seed, fold), f in both.groupby(["seed", "fold"], sort=True):
        pos = f["label"].to_numpy(int) == 1
        if not pos.any():
            continue  # sin positivos no hay TPR que comparar
        tpr = {
            m: float(hit_probability_at_ppr(f[f"score_{m}"].to_numpy(float), ppr)[pos].mean())
            for m in ("a", "b")
        }
        rows.append({"seed": seed, "fold": fold, "n_test": len(f),
                     "n_common": int(n_common[seed]), "n_pos": int(pos.sum()),
                     "tpr_a": tpr["a"], "tpr_b": tpr["b"],
                     "diff": tpr["a"] - tpr["b"]})  # fmt: skip
    if not rows:
        raise ValueError(f"Sin folds comparables en {scheme}.")
    return pd.DataFrame(rows)


def _check_once_per_seed(d: pd.DataFrame, what: str) -> None:
    """Cada positivo debe evaluarse exactamente una vez por semilla (preregistro §8)."""
    counts = d.groupby(["composition", "seed"]).size()
    if (counts != 1).any():
        raise ValueError(f"{what}: hay positivos evaluados más de una vez en una semilla.")
    per_comp = d.groupby("composition")["seed"].nunique()
    if per_comp.nunique() != 1:
        raise ValueError(f"{what}: no todos los positivos se evalúan en todas las semillas.")


# ---------------------------------------------------------------------------
# Unidades del bootstrap
# ---------------------------------------------------------------------------


def bootstrap_units(
    compositions: pd.DataFrame, scheme: str, b_families: Mapping[str, str] | None = None
) -> tuple[pd.Series, pd.Series | None]:
    """Unidad química de nivel 1 (y estrato) de cada composición para un esquema (§9).

    `compositions` tiene índice de composición y columnas `A` y `B`. Elemento A en `loeo_A`;
    elemento B en `loeo_B`, `family` y `cluster`. En `family`, los elementos B se remuestrean
    dentro de su familia (`b_families`: elemento B → familia), porque con 4 familias no se
    pueden remuestrear las familias mismas.
    """
    if scheme == "loeo_A":
        return compositions["A"].rename("unit"), None
    if scheme not in ("loeo_B", "family", "cluster"):
        raise ValueError(f"El bootstrap jerárquico no está definido para `{scheme}`.")
    units = compositions["B"].rename("unit")
    if scheme != "family":
        return units, None
    if b_families is None:
        raise ValueError("`family` necesita la familia de cada elemento B.")
    strata = units.map(b_families)
    if strata.isna().any():
        raise ValueError(f"Elementos B sin familia: {sorted(units[strata.isna()].unique())}")
    return units, strata.rename("stratum")


def bootstrap_rng(seed: int, hypothesis: str, scheme: str) -> np.random.Generator:
    """Flujo aleatorio de un bootstrap, derivado de `bootstrap_seed` (configs/seeds.yaml).

    Cada (hipótesis, esquema) tiene su propio flujo independiente y reproducible, aunque se
    calculen en otro orden o por separado.
    """
    key = (HYPOTHESES.index(hypothesis), BOOTSTRAP_SCHEMES.index(scheme))
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=key))


# ---------------------------------------------------------------------------
# Bootstrap jerárquico
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapResult:
    """Estimación puntual, IC por percentiles y valores p del bootstrap (§9)."""

    estimate: float  # media de d(c) sobre todas las composiciones
    ci_low: float
    ci_high: float
    p_two_sided: float  # 2·min{P*(Δ* ≤ 0), P*(Δ* ≥ 0)}, acotado a 1
    p_le_margin: float  # P*(Δ* ≤ margin): unilateral (H2: margin = −δ₂)
    p_le_zero: float  # P*(Δ* ≤ 0): unilateral de superioridad
    margin: float
    n_units: int
    n_compositions: int
    n_boot: int

    def as_dict(self) -> dict[str, float]:
        """Diccionario plano para tablas."""
        return asdict(self)


def hierarchical_bootstrap_distribution(
    d: Sequence[float],
    units: Sequence[str],
    strata: Sequence[str] | None = None,
    n_boot: int = 10_000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Distribución bootstrap de la media de d(c) con remuestreo en dos niveles.

    Nivel 1: unidades con reemplazo (dentro de cada estrato si se da `strata`, conservando el
    número de unidades del estrato). Nivel 2: composiciones con reemplazo dentro de cada unidad
    elegida, con un remuestreo independiente cada vez que la unidad sale. El estadístico es la
    media sobre todas las composiciones remuestreadas (ponderación por composición, como la
    TPR micro-agregada).
    """
    rng = rng if rng is not None else np.random.default_rng()
    d = np.asarray(d, dtype=float)
    units = np.asarray(units)
    if d.shape != units.shape or d.ndim != 1 or not d.size:
        raise ValueError("`d` y `units` deben ser vectores de la misma longitud, no vacíos.")
    if np.isnan(d).any():
        raise ValueError("`d` contiene NaN.")
    # Composiciones agrupadas por unidad: valores contiguos, con desplazamiento y tamaño.
    unit_ids, inverse = np.unique(units, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    values = d[order]
    sizes = np.bincount(inverse)
    offsets = np.r_[0, np.cumsum(sizes)[:-1]]
    # Unidades de cada estrato (un único estrato si no se estratifica).
    if strata is None:
        groups = [np.arange(unit_ids.size)]
    else:
        strata = np.asarray(strata)
        if strata.shape != units.shape:
            raise ValueError("`strata` debe tener la misma longitud que `units`.")
        unit_stratum = pd.Series(strata).groupby(inverse).agg(lambda s: s.unique())
        if any(len(s) != 1 for s in unit_stratum):
            raise ValueError("Una unidad pertenece a más de un estrato.")
        stratum_of_unit = np.array([s[0] for s in unit_stratum])
        groups = [np.flatnonzero(stratum_of_unit == s) for s in np.unique(stratum_of_unit)]
    out = np.empty(n_boot)
    for b in range(n_boot):
        # Nivel 1: unidades elegidas (con reemplazo, dentro de cada estrato).
        chosen = np.concatenate([rng.choice(g, size=g.size, replace=True) for g in groups])
        n = sizes[chosen]
        # Nivel 2: n composiciones con reemplazo dentro de cada unidad elegida.
        start = np.repeat(offsets[chosen], n)
        pick = start + (rng.random(n.sum()) * np.repeat(n, n)).astype(int)
        out[b] = values[pick].mean()
    return out


def summarize_bootstrap(
    estimate: float,
    dist: np.ndarray,
    margin: float = 0.0,
    ci: float = 0.95,
    n_units: int = 0,
    n_compositions: int = 0,
) -> BootstrapResult:
    """IC por percentiles y valores p (§9) a partir de la distribución bootstrap."""
    alpha = 1.0 - ci
    lo, hi = np.quantile(dist, [alpha / 2, 1 - alpha / 2])
    p_le0, p_ge0 = float(np.mean(dist <= 0)), float(np.mean(dist >= 0))
    return BootstrapResult(
        estimate=float(estimate),
        ci_low=float(lo),
        ci_high=float(hi),
        p_two_sided=min(1.0, 2 * min(p_le0, p_ge0)),
        p_le_margin=float(np.mean(dist <= margin)),
        p_le_zero=p_le0,
        margin=float(margin),
        n_units=int(n_units),
        n_compositions=int(n_compositions),
        n_boot=int(dist.size),
    )


def hierarchical_bootstrap(
    d: pd.Series,
    units: pd.Series,
    strata: pd.Series | None = None,
    margin: float = 0.0,
    n_boot: int = 10_000,
    rng: np.random.Generator | None = None,
    ci: float = 0.95,
) -> BootstrapResult:
    """Bootstrap jerárquico de la media de d(c), alineando por composición (índice)."""
    if d.index.has_duplicates:
        raise ValueError("`d` tiene composiciones repetidas.")
    missing = d.index.difference(units.index)
    if len(missing):
        raise ValueError(f"Composiciones sin unidad: {sorted(missing)[:5]}")
    u = units.reindex(d.index)
    s = strata.reindex(d.index) if strata is not None else None
    dist = hierarchical_bootstrap_distribution(
        d.to_numpy(float), u.to_numpy(), None if s is None else s.to_numpy(), n_boot, rng
    )
    return summarize_bootstrap(
        float(d.mean()), dist, margin, ci, n_units=u.nunique(), n_compositions=d.size
    )


# ---------------------------------------------------------------------------
# Multiplicidad y t corregido
# ---------------------------------------------------------------------------


def holm(pvalues: Sequence[float]) -> np.ndarray:
    """Valores p ajustados por Holm (paso descendente, monótonos, acotados a 1)."""
    p = np.asarray(pvalues, dtype=float)
    if np.isnan(p).any():
        raise ValueError("Holm no admite valores p NaN.")
    m = p.size
    order = np.argsort(p, kind="stable")
    adjusted = np.maximum.accumulate((m - np.arange(m)) * p[order])
    out = np.empty(m)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def nadeau_bengio(
    diffs: Sequence[float], n_train: float, n_test: float, ci: float = 0.95
) -> dict[str, float]:
    """t de Student corregido de Nadeau y Bengio (2003) para CV repetida.

    `diffs` son las diferencias de la métrica entre dos modelos en cada fold de cada repetición
    (J = r · k valores). La varianza se corrige por el solapamiento de los conjuntos de
    entrenamiento: (1/J + n_test/n_train) · s². Grados de libertad J − 1. Con tamaños de fold
    distintos, pasar las medias de n_train y n_test.
    """
    d = np.asarray(diffs, dtype=float)
    j = d.size
    if j < 2:
        raise ValueError("Se necesitan al menos 2 diferencias.")
    mean, var = float(d.mean()), float(d.var(ddof=1))
    se = np.sqrt((1.0 / j + n_test / n_train) * var)
    df = j - 1
    # Diferencias idénticas: la varianza es 0 salvo error de coma flotante (≈ 1e-33).
    if np.ptp(d) == 0:
        # Todas las diferencias iguales: sin variabilidad, el test es degenerado.
        t = 0.0 if mean == 0 else np.copysign(np.inf, mean)
        p = 1.0 if mean == 0 else 0.0
        return {"mean": mean, "t": t, "df": df, "p_two_sided": p, "ci_low": mean,
                "ci_high": mean, "se": 0.0, "n_diffs": j}  # fmt: skip
    t = mean / se
    p = float(2 * sps.t.sf(abs(t), df))
    q = sps.t.ppf(1 - (1 - ci) / 2, df)
    return {"mean": mean, "t": float(t), "df": df, "p_two_sided": p,
            "ci_low": mean - q * se, "ci_high": mean + q * se, "se": float(se),
            "n_diffs": j}  # fmt: skip


# ---------------------------------------------------------------------------
# Hipótesis confirmatorias (§8)
# ---------------------------------------------------------------------------


def _table(results: Mapping[str, BootstrapResult]) -> pd.DataFrame:
    """Una fila por esquema con los campos del bootstrap."""
    return pd.DataFrame({k: v.as_dict() for k, v in results.items()}).T.rename_axis("scheme")


def h1_decision(results: Mapping[str, BootstrapResult], delta1: float = 0.05,
                alpha: float = 0.05) -> pd.DataFrame:  # fmt: skip
    """H1: caída relevante de TPR al pasar de `random` a cada esquema por grupos.

    Cada resultado es el bootstrap de d(c) = h_random(c) − h_g(c). Criterio por esquema: valor
    p bilateral ajustado por Holm < α **y** ΔTPR ≥ δ₁. Global: los 4 esquemas (columna
    `global` repetida en todas las filas).
    """
    t = _table(results)
    t["p_holm"] = holm(t["p_two_sided"].to_numpy(float))
    t["criterion"] = (t["p_holm"] < alpha) & (t["estimate"].astype(float) >= delta1)
    t["global"] = bool(t["criterion"].all())
    return t


def h2_decision(results: Mapping[str, BootstrapResult], delta2: float = 0.05,
                alpha_one_sided: float = 0.025) -> pd.DataFrame:  # fmt: skip
    """H2: no inferioridad del PU frente a τ a igual P(ŷ = 1), con margen δ₂.

    Cada resultado es el bootstrap de d(c) = h_PU(c) − h_τ(c) con `margin = −δ₂`. No
    inferioridad: Holm sobre P*(Δ* ≤ −δ₂) < 0,025 (equivale a que la cota inferior del IC 95 %
    bilateral supere −δ₂). Superioridad (secundaria): Holm sobre P*(Δ* ≤ 0) < 0,025.
    """
    t = _table(results)
    if not np.allclose(t["margin"].astype(float), -delta2):
        raise ValueError("Los bootstraps de H2 deben calcularse con margin = −δ₂.")
    t["p_noninf_holm"] = holm(t["p_le_margin"].to_numpy(float))
    t["noninferior"] = t["p_noninf_holm"] < alpha_one_sided
    t["p_sup_holm"] = holm(t["p_le_zero"].to_numpy(float))
    t["superior"] = t["noninferior"] & (t["p_sup_holm"] < alpha_one_sided)
    t["global_noninferior"] = bool(t["noninferior"].all())
    return t


def h3_decision(results: Mapping[str, BootstrapResult], alpha: float = 0.05) -> pd.DataFrame:
    """H3: la ventaja del grafo sobre el PU composicional se reduce al extrapolar.

    Cada resultado es el bootstrap de D(c) = [h_graf,random − h_comp,random] −
    [h_graf,g − h_comp,g]. Criterio por esquema: D > 0 y valor p bilateral (Holm) < α.
    """
    t = _table(results)
    t["p_holm"] = holm(t["p_two_sided"].to_numpy(float))
    t["criterion"] = (t["p_holm"] < alpha) & (t["estimate"].astype(float) > 0)
    t["global"] = bool(t["criterion"].all())
    return t
