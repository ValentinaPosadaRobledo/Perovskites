"""Figura: caída de TPR de H1 en cada análisis de sensibilidad (F3.6), pareada con F3.3.

Entrada: `results/f3_6_sensibilidad/paired.csv` (`scripts/report_f3_6.py`). Un panel por
esquema confirmatorio; una fila por sensibilidad. Punto lleno = caída de `random` al esquema en
la sensibilidad; punto vacío gris = la misma caída en el análisis principal (F3.3) sobre las
MISMAS positivas; la línea los une. Línea discontinua en δ₁ = 0,05. Medias de 10 semillas;
descriptivo (§10 no preregistra pruebas para las sensibilidades).

Quedan fuera la 6 y la 12 (solo cambian `loeo_B` / `loeo_A`, sin caída frente a `random`), y la
variante de control a recall 0,90 de la 8, que reproduce F3.3 exactamente. En la 4 la
referencia es `post_dft` con GGA/GGA+U en las mismas 173 composiciones, no el PU `pre_dft`.
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

NAME = "fig_sensitivity_h1"
SRC = ROOT / "results" / "f3_6_sensibilidad" / "paired.csv"
DELTA1 = 0.05  # preregistro §8
# (sensibilidad, comparación en paired.csv, etiqueta en inglés, etiqueta en español)
ROWS = [
    ("f3_6_s01_high_pressure", "PU pre_dft", "1  no high-pressure/metastable",
     "1  sin alta presión/metaestables"),
    ("f3_6_s02_ambiguous_oxidation", "PU pre_dft", "2  no ambiguous oxidation",
     "2  sin oxidación ambigua"),
    ("f3_6_s04_r2scan", "post_dft r2SCAN frente a GGA", "4  r2SCAN (post-DFT)",
     "4  r2SCAN (post-DFT)"),
    ("f3_6_s05_angle140", "PU pre_dft", "5  B–O–B ≥ 140°", "5  B–O–B ≥ 140°"),
    ("f3_6_s08_recall", "recall 0,80", "8  recall 0.80", "8  recall 0,80"),
    ("f3_6_s08_recall", "recall 0,95", "8  recall 0.95", "8  recall 0,95"),
    ("f3_6_s11_charge_balanced_u", "PU pre_dft", "11  charge-balanced U only",
     "11  solo U con balance de carga"),
    ("f3_6_s13_ambiguous_as_positive", "PU pre_dft", "13  ambiguous as P",
     "13  ambiguas como P"),
    ("f3_6_s14_pre_dft_tau", "todas (τ in-sample en la TableS1)", "14  + τ (all)",
     "14  + τ (todas)"),
    ("f3_6_s14_pre_dft_tau", "solo fuera de la TableS1", "14  + τ (outside TableS1, 41 P)",
     "14  + τ (fuera de la TableS1, 41 P)"),
]  # fmt: skip
TEXT = {
    "en": {"x": "TPR drop from random split", "sens": "sensitivity",
           "ref": "main analysis, same positives", "d1": "δ₁"},
    "es": {"x": "Caída de TPR desde la partición aleatoria", "sens": "sensibilidad",
           "ref": "análisis principal, mismas positivas", "d1": "δ₁"},
}  # fmt: skip


def draw(paired: pd.DataFrame, lang: str) -> plt.Figure:
    """Un panel por esquema confirmatorio; filas = sensibilidades, de arriba abajo."""
    t = TEXT[lang]
    fig, axes = plt.subplots(1, len(CONFIRMATORY), figsize=(7.2, 3.6), sharey=True)
    y = np.arange(len(ROWS))[::-1]
    for ax, scheme in zip(axes, CONFIRMATORY, strict=True):
        ax.axvline(0, color=INK["secondary"], lw=0.7, zorder=1)
        ax.axvline(DELTA1, color=INK["muted"], lw=0.8, ls=(0, (4, 3)), zorder=1)
        for yi, (sens, comp, _, _) in zip(y, ROWS, strict=True):
            r = paired[(paired["sensitivity"] == sens) & (paired["comparison"] == comp)
                       & (paired["scheme"] == scheme)]  # fmt: skip
            if r.empty:  # la 13 no tiene `temporal`; aquí no falta ningún esquema confirmatorio
                continue
            s, ref = float(r["h1_drop_sens"].iloc[0]), float(r["h1_drop_ref_same"].iloc[0])
            ax.plot([ref, s], [yi, yi], color=INK["grid"], lw=1.6, zorder=2,
                    solid_capstyle="round")  # fmt: skip
            ax.plot(ref, yi, "o", ms=4.5, mfc="#ffffff", mec=INK["muted"], mew=1.0, zorder=3,
                    label=t["ref"] if yi == y[0] else None)  # fmt: skip
            ax.plot(s, yi, "o", ms=4.5, color=SERIES[0], mec="#ffffff", mew=0.6, zorder=4,
                    label=t["sens"] if yi == y[0] else None)  # fmt: skip
        ax.set_title(SCHEME_LABEL[lang][scheme])
        ax.set_xlim(-0.05, 0.33)
        ax.grid(axis="y", visible=False)
        ax.text(DELTA1 + 0.006, 0.005, t["d1"], ha="left", va="bottom", color=INK["secondary"],
                fontsize=7, transform=ax.get_xaxis_transform())  # fmt: skip
    axes[0].set_yticks(y, [row[2] if lang == "en" else row[3] for row in ROWS])
    axes[0].set_ylim(-1.0, len(ROWS) - 0.5)  # hueco bajo la última fila para la etiqueta δ₁
    axes[0].tick_params(axis="y", length=0)
    fig.supxlabel(t["x"], fontsize=8, color=INK["secondary"], y=0.075)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", ncol=2, bbox_to_anchor=(0.99, 0.0),
               handletextpad=0.3)  # fmt: skip
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=0.8)
    return fig


def main() -> None:
    setup_style()
    paired = pd.read_csv(SRC)
    for lang in TEXT:
        save(draw(paired, lang), NAME, lang)
    note = "Medias de 10 semillas, pareadas por composición con F3.3; descriptivo."
    write_provenance(FIG_DIR / f"{NAME}.json", [SRC], note)


if __name__ == "__main__":
    main()
