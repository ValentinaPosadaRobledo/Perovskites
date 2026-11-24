"""Figura (tesis, OE4): frente de Pareto sintetizabilidad–estabilidad acuosa (F5.2).

Entrada: `candidates.csv` de la última corrida de `f5_2_pareto` (condición principal: pH 7,4 e
iones a 1e-6 M). Eje x: ΔG_pbx máximo en la ventana de −0,2 a +0,4 V (eV/átomo); eje y:
puntuación PU `pre_dft` fuera de fold en `random` (media de 10 semillas).

Series: no etiquetadas elegibles (sin marcas de toxicidad), no etiquetadas marcadas (fuera del
frente) y el frente principal (rango 1), con su nombre. Línea discontinua: la referencia de
~0,5 eV/átomo de Singh et al. (2017). Las positivas no se dibujan: no son candidatas.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import FIG_DIR, INK, ROOT, SERIES, plt, save, setup_style, write_provenance

NAME = "fig_pareto_pourbaix"
REF = 0.5  # Singh et al. (2017)
TEXT = {
    "en": {"x": "ΔG_pbx, worst case in −0.2 to +0.4 V (eV/atom)",
           "y": "PU pre-DFT score (out of fold, random)",
           "elig": "unlabeled, no toxicity flag", "tox": "unlabeled, flagged (excluded)",
           "front": "Pareto front", "ref": "~0.5 eV/atom"},
    "es": {"x": "ΔG_pbx, peor caso en −0,2 a +0,4 V (eV/átomo)",
           "y": "Puntuación PU pre-DFT (fuera de fold, aleatoria)",
           "elig": "no etiquetadas, sin marca de toxicidad",
           "tox": "no etiquetadas, marcadas (excluidas)",
           "front": "frente de Pareto", "ref": "~0,5 eV/átomo"},
}  # fmt: skip


def candidates_path() -> Path:
    """`candidates.csv` de la última corrida de F5.2 registrada en `runs_index.csv`."""
    index = pd.read_csv(ROOT / "results" / "runs_index.csv")
    return ROOT / index[index["experiment"] == "f5_2_pareto"].iloc[-1]["run_dir"] / "candidates.csv"


def draw(c: pd.DataFrame, lang: str) -> plt.Figure:
    """Dispersión de las no etiquetadas con el frente resaltado y nombrado."""
    t = TEXT[lang]
    u = c[c["label"] == 0]
    tox = u["toxicity"].fillna("") != ""
    front = u[u["on_front"].astype(bool)].sort_values("dg_max")
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.axvline(REF, color=INK["muted"], lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.text(REF, 1.01, t["ref"], color=INK["secondary"], fontsize=7, va="bottom", ha="center",
            transform=ax.get_xaxis_transform())  # fmt: skip
    ax.plot(u.loc[tox, "dg_max"], u.loc[tox, "score_mean"], "x", ms=3.5, mew=0.8,
            color=INK["muted"], label=t["tox"], zorder=2)  # fmt: skip
    ax.plot(u.loc[~tox, "dg_max"], u.loc[~tox, "score_mean"], "o", ms=3.2, color=SERIES[0],
            alpha=0.55, mec="none", label=t["elig"], zorder=3)  # fmt: skip
    ax.plot(front["dg_max"], front["score_mean"], "-", color=SERIES[1], lw=1.0, zorder=4,
            drawstyle="steps-post")  # fmt: skip
    ax.plot(front["dg_max"], front["score_mean"], "o", ms=6, color=SERIES[1], mec="#ffffff",
            mew=0.8, label=t["front"], zorder=5)  # fmt: skip
    for _, r in front.iterrows():
        ax.annotate(r["composition"], (r["dg_max"], r["score_mean"]), xytext=(7, 0),
                    textcoords="offset points", fontsize=7, color=INK["primary"], va="center",
                    bbox={"boxstyle": "round,pad=0.15", "fc": "#ffffff", "ec": "none",
                          "alpha": 0.85}, zorder=6)  # fmt: skip
    ax.set_xlabel(t["x"])
    ax.set_ylabel(t["y"])
    ax.set_xlim(left=-0.03)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper right", handletextpad=0.3)
    return fig


def main() -> None:
    setup_style()
    path = candidates_path()
    c = pd.read_csv(path)
    for lang in TEXT:
        save(draw(c, lang), NAME, lang)
    note = "Condición principal (pH 7,4; 1e-6 M); frente de las no etiquetadas sin marcas."
    write_provenance(FIG_DIR / f"{NAME}.json", [path], note)


if __name__ == "__main__":
    main()
