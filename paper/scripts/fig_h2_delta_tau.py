"""Figura: H2, PU `pre_dft` frente a τ a igual P(ŷ = 1) (descriptivo; la prueba es de F4.2).

Entrada: `results/f3_3_pu/h2_descriptive.csv` (por esquema y semilla). En cada fold, el PU se
umbraliza a la P(ŷ = 1) que da τ < 4,18 en las composiciones con τ definido (§7.2);
Δ = TPR_PU − TPR_τ. Punto = media de las 10 semillas; barra = mínimo–máximo entre semillas.
Línea discontinua = margen de no inferioridad −δ₂ = −0,05. Dos series: análisis principal (con
los compuestos de Bartel, donde τ es *in-sample*) y sensibilidad sin ellos (τ fuera de muestra).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    CONFIRMATORY,
    FIG_DIR,
    INK,
    ROOT,
    SCHEME_LABEL,
    SERIES,
    plt,
    save,
    setup_style,
    write_provenance,
)

NAME = "fig_h2_delta_tau"
SRC = ROOT / "results" / "f3_3_pu" / "h2_descriptive.csv"
ORDER = ["random", "loeo_A", "loeo_B", "family", "cluster"]  # temporal: test de 27 P
TEXT = {
    "en": {"y": "ΔTPR, PU − τ at τ's P(ŷ = 1)", "margin": "non-inferiority margin −δ₂",
           "principal": "with Bartel's compounds (τ in-sample)",
           "sin_bartel2019": "without them (τ out of sample)", "conf": "confirmatory"},
    "es": {"y": "ΔTPR, PU − τ a la P(ŷ = 1) de τ", "margin": "margen de no inferioridad −δ₂",
           "principal": "con los compuestos de Bartel (τ in-sample)",
           "sin_bartel2019": "sin ellos (τ fuera de muestra)", "conf": "confirmatorios"},
}  # fmt: skip


def draw(h2: pd.DataFrame, lang: str) -> plt.Figure:
    """Media y rango entre semillas de Δ por esquema, para las dos variantes del análisis."""
    t = TEXT[lang]
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    x = np.arange(len(ORDER))
    i0, i1 = ORDER.index(CONFIRMATORY[0]), ORDER.index(CONFIRMATORY[-1])
    ax.axvspan(i0 - 0.45, i1 + 0.45, color="#f3f2ef", zorder=0.1, lw=0)
    ax.text((i0 + i1) / 2, 0.985, t["conf"], ha="center", va="top", color=INK["secondary"],
            fontsize=7, transform=ax.get_xaxis_transform())  # fmt: skip
    ax.axhline(0, color=INK["secondary"], lw=0.7, zorder=1)
    ax.axhline(-0.05, color=INK["muted"], lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.text(-0.5, -0.048, t["margin"], ha="left", va="bottom", color=INK["secondary"],
            fontsize=7)  # fmt: skip
    for k, analysis in enumerate(("principal", "sin_bartel2019")):
        g = h2[h2["analysis"] == analysis].groupby("scheme")["delta"]
        mean, lo, hi = (g.mean().reindex(ORDER), g.min().reindex(ORDER), g.max().reindex(ORDER))
        xs = x + (k - 0.5) * 0.22
        ax.errorbar(xs, mean, yerr=np.vstack([mean - lo, hi - mean]), fmt="o", ms=4.5,
                    color=SERIES[k], lw=1.2, capsize=0, mec="#ffffff", mew=0.8,
                    label=t[analysis], zorder=3)  # fmt: skip
    ax.set_xticks(x, [SCHEME_LABEL[lang][s] for s in ORDER])
    ax.set_ylabel(t["y"])
    ax.set_xlim(-0.6, len(ORDER) - 0.4)
    ax.set_ylim(-0.25, 0.12)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left", handletextpad=0.3)
    return fig


def main() -> None:
    setup_style()
    h2 = pd.read_csv(SRC)
    for lang in TEXT:
        save(draw(h2, lang), NAME, lang)
    write_provenance(FIG_DIR / f"{NAME}.json", [SRC],
                     "Media y mínimo–máximo de Δ entre 10 semillas; descriptivo.")  # fmt: skip


if __name__ == "__main__":
    main()
