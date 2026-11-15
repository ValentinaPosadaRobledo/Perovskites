"""F3.4: conjuntos y métricas de la validación de **formabilidad** (no de sintetizabilidad).

Los negativos son óxidos ABO₃ que se han sintetizado y NO son perovskita: los de nuestra
etiquetación (`synthesized_non_perovskite`, F1.5) y los negativos experimentales de la TableS1
de Bartel et al. (2019). Un modelo de sintetizabilidad no tiene por qué rechazarlos (son
sintetizables), pero un buen modelo de "sintetizable COMO perovskita" debería puntuarlos por
debajo de las perovskitas conocidas. Por eso aquí se mide formabilidad: FPR sobre estos
negativos y ROC-AUC real entre P y negativos (hay negativos verdaderos, no son cotas).

Conjuntos (decisión de la autora, 2026-11-14):

- `principal`: sintetizadas no perovskita (nuestras ∪ Bartel), sin polianiones.
- `polyanion`: negativos de Bartel que nosotros clasificamos como polianiones; fuera del
  dominio del modelo (regla 2.1.1), se reportan aparte.
- `ambiguous`: negativos de Bartel que nosotros dejamos como ambiguos; aparte.
- `conflict_positive` / `conflict_unlabeled`: negativos de Bartel que para nosotros son P o U
  (están en train_pool): no se puntúan; solo se cuentan.
- `excluded_reserved_b`: con el elemento B en el conjunto reservado; no se puntúan nunca.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from pymatgen.core import Composition
from sklearn.metrics import roc_auc_score

SCORED_SETS = ("principal", "polyanion", "ambiguous")
UNSCORED_SETS = ("conflict_positive", "conflict_unlabeled", "excluded_reserved_b")


def formability_sets(
    labels: pd.DataFrame,
    table_s1: pd.DataFrame,
    reserved_b: Iterable[str],
    pool_compositions: Iterable[str],
) -> pd.DataFrame:
    """Una fila por candidato a negativo de formabilidad, con su conjunto y su procedencia.

    `labels`: etiquetas por composición de F1.5 (`reduced_formula`, `label`, `A`, `B`).
    `table_s1`: TableS1 de Bartel et al. (2019) (`ABX3`, `exp_label`, `is_train`, `A`, `B`, `X`).
    """
    reserved = set(reserved_b)
    lab = labels.set_index("reduced_formula")
    ox = table_s1[table_s1["X"] == "O"].copy()
    ox["formula"] = [Composition(f).reduced_formula for f in ox["ABX3"]]
    bneg = ox[ox["exp_label"] == -1].drop_duplicates("formula").set_index("formula")
    ours = set(lab.index[lab["label"] == "synthesized_non_perovskite"])
    rows = []
    for f in sorted(ours | set(bneg.index)):
        our = lab["label"].get(f)
        a = lab["A"].get(f) if f in lab.index else bneg.at[f, "A"]
        b = lab["B"].get(f) if f in lab.index else bneg.at[f, "B"]
        in_s1 = f in bneg.index
        if b in reserved:
            kind = "excluded_reserved_b"  # nunca se puntúa (elemento B reservado)
        elif our == "positive":
            kind = "conflict_positive"
        elif our == "unlabeled":
            kind = "conflict_unlabeled"
        elif our == "polyanion":
            kind = "polyanion"
        elif our == "ambiguous":
            kind = "ambiguous"
        else:  # sintetizada no perovskita para nosotros, o fuera de nuestro espacio
            kind = "principal"
        rows.append({
            "composition": f, "A": a, "B": b, "set": kind,
            "source": "both" if (f in ours and in_s1) else ("ours" if f in ours else "bartel2019"),
            "our_label": our if our is not None else "absent",
            "in_bartel2019": in_s1,
            "bartel2019_is_train": bool(bneg.at[f, "is_train"]) if in_s1 else False,
        })  # fmt: skip
    out = pd.DataFrame(rows)
    # Garantía: ningún negativo puntuado está en train_pool (ni como P ni como U).
    scored = set(out.loc[out["set"].isin(SCORED_SETS), "composition"])
    overlap = scored & set(pool_compositions)
    if overlap:
        raise AssertionError(f"Negativos de formabilidad en train_pool: {sorted(overlap)[:5]}")
    return out


def positive_rate(pred: np.ndarray) -> float:
    """Fracción de predicciones positivas (TPR en P, P(ŷ=1|U) en U, FPR en negativos)."""
    pred = np.asarray(pred)
    return float(pred.mean()) if pred.size else float("nan")


def auc_positive_vs_negative(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    """ROC-AUC entre positivas conocidas y negativos verdaderos (no es una cota inferior)."""
    scores_pos, scores_neg = np.asarray(scores_pos, float), np.asarray(scores_neg, float)
    if not scores_pos.size or not scores_neg.size:
        return float("nan")
    y = np.r_[np.ones(scores_pos.size), np.zeros(scores_neg.size)]
    return float(roc_auc_score(y, np.r_[scores_pos, scores_neg]))
