"""F5.2: filtro de toxicidad y frente de Pareto sintetizabilidad–estabilidad acuosa (OE4).

- **Sintetizabilidad:** puntuación fuera de fold del PU `pre_dft` en el esquema `random` de F3.3
  (cada composición se puntúa una vez por semilla, en el fold donde es de test): media y DE
  entre semillas y fracción de semillas en que supera el umbral del bucle interno.
- **Estabilidad acuosa:** ΔG_pbx máximo (peor caso) en la ventana de potencial, de F5.1.
- **Toxicidad:** Pb, Co, Ni, Cd y Tl por presencia; Ba²⁺ soluble y Cr⁶⁺ si aparecen entre las
  especies estables de la ventana en la condición evaluada.
- **Frente de Pareto:** ordenación no dominada (máxima puntuación, mínimo ΔG_pbx) de las no
  etiquetadas sin marcas de toxicidad; rango 1 = frente. Las P y las marcadas se informan con
  su posición, pero no entran en el frente.

Descriptivo y exploratorio (solo tesis).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def oof_scores(raw: pd.DataFrame, model: str, scheme: str) -> pd.DataFrame:
    """Puntuación fuera de fold por composición: media, DE y fracción sobre el umbral.

    Raises:
        ValueError: si alguna composición no se puntúa exactamente una vez por semilla.
    """
    d = raw[(raw["baseline"] == model) & (raw["scheme"] == scheme)]
    counts = d.groupby(["composition", "seed"]).size()
    if d.empty or (counts != 1).any() or d.groupby("composition")["seed"].nunique().nunique() != 1:
        raise ValueError(f"{model}/{scheme}: cada composición debe puntuarse una vez por semilla.")
    g = d.groupby("composition")
    return pd.DataFrame({"score_mean": g["score"].mean(), "score_sd": g["score"].std(ddof=1),
                         "frac_above_threshold": g["pred"].mean(),
                         "n_seeds": g["seed"].nunique()}).reset_index()  # fmt: skip


def toxicity_flags(
    table: pd.DataFrame, elements: Sequence[str], species: Mapping[str, str]
) -> pd.Series:
    """Motivos de toxicidad de cada fila (texto vacío si no hay ninguno).

    `table` tiene las columnas `A`, `B` y `stable_in_window` (especies estables de la ventana
    separadas por " | ", de F5.1).
    """
    pats = {name: re.compile(p) for name, p in species.items()}
    out = []
    for a, b, window in zip(table["A"], table["B"], table["stable_in_window"].fillna(""),
                            strict=True):  # fmt: skip
        why = [el for el in elements if el in (a, b)]
        why += [name for name, pat in pats.items() if pat.search(window)]
        out.append(";".join(why))
    return pd.Series(out, index=table.index, name="toxicity")


def pareto_ranks(maximize: np.ndarray, minimize: np.ndarray) -> np.ndarray:
    """Rango de Pareto (1 = no dominado) por ordenación no dominada sucesiva.

    Un punto domina a otro si no es peor en ninguno de los dos objetivos y es estrictamente
    mejor en alguno. Los puntos idénticos comparten rango.
    """
    x, y = np.asarray(maximize, float), np.asarray(minimize, float)
    n = x.size
    ranks = np.zeros(n, dtype=int)
    remaining = np.arange(n)
    r = 0
    while remaining.size:
        r += 1
        xs, ys = x[remaining], y[remaining]
        dominated = np.zeros(remaining.size, dtype=bool)
        for i in range(remaining.size):
            better_eq = (xs >= xs[i]) & (ys <= ys[i])
            strictly = (xs > xs[i]) | (ys < ys[i])
            dominated[i] = bool(np.any(better_eq & strictly))
        ranks[remaining[~dominated]] = r
        remaining = remaining[dominated]
    return ranks


def condition_front(
    table: pd.DataFrame, ph: float, conc: float, elements: Sequence[str],
    species: Mapping[str, str],
) -> pd.DataFrame:  # fmt: skip
    """Candidatas, marcas y rango de Pareto en una condición (pH, concentración).

    `table` une F5.1 (una fila por composición × pH × concentración) con las puntuaciones.
    El rango solo se asigna a las no etiquetadas sin marcas (el resto queda en 0).
    """
    sub = table[np.isclose(table["pH"], ph) & np.isclose(table["ion_conc_M"], conc)].copy()
    sub["toxicity"] = toxicity_flags(sub, elements, species)
    eligible = (sub["label"] == 0) & (sub["toxicity"] == "")
    sub["pareto_rank"] = 0
    x = sub.loc[eligible, "score_mean"].to_numpy()
    sub.loc[eligible, "pareto_rank"] = pareto_ranks(x, sub.loc[eligible, "dg_max"].to_numpy())
    sub["on_front"] = sub["pareto_rank"] == 1
    return sub
