"""Figura: TPR en el punto de operación (recall nominal 0,90) por esquema de partición (H1).

Entrada: `results/f3_3_pu/summary.csv` (10 semillas). Puntos = media; barras = IC 95 %
(t de Student entre semillas). Línea discontinua = recall nominal del bucle interno: la caída
por debajo mide cuánto se degrada un umbral calibrado al exigir extrapolación. Sombreado = los
4 esquemas confirmatorios. Dos series: `pre_dft` (principal) y `post_dft` (cota superior
informada por el hull).
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
    SCHEMES,
    SERIES,
    plt,
    save,
    setup_style,
    write_provenance,
)

NAME = "fig_tpr_by_scheme"
SRC = ROOT / "results" / "f3_3_pu" / "summary.csv"
TEXT = {
    "en": {"y": "TPR at the operating point", "nominal": "nominal recall (inner loop) 0.90",
           "conf": "confirmatory", "pre": "PU pre-DFT (composition)",
           "post": "PU post-DFT (+ E$_{hull}$, ΔH$_f$)"},
    "es": {"y": "TPR en el punto de operación", "nominal": "recall nominal (bucle interno) 0,90",
           "conf": "confirmatorios", "pre": "PU pre-DFT (composición)",
           "post": "PU post-DFT (+ E$_{hull}$, ΔH$_f$)"},
}  # fmt: skip


def draw(summary: pd.DataFrame, lang: str) -> plt.Figure:
    """Puntos con IC 95 % por esquema, dos modelos desplazados horizontalmente."""
    t = TEXT[lang]
    fig, ax = plt.subplots(figsize=(6.3, 2.8))
    x = np.arange(len(SCHEMES))
    i0, i1 = SCHEMES.index(CONFIRMATORY[0]), SCHEMES.index(CONFIRMATORY[-1])
    ax.axvspan(i0 - 0.45, i1 + 0.45, color="#f3f2ef", zorder=0.1, lw=0)
    ax.text((i0 + i1) / 2, 0.985, t["conf"], ha="center", va="top", color=INK["secondary"],
            fontsize=7, transform=ax.get_xaxis_transform())  # fmt: skip
    ax.axhline(0.90, color=INK["muted"], lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.text(len(SCHEMES) - 0.5, 0.902, t["nominal"], ha="right", va="bottom",
            color=INK["secondary"], fontsize=7)  # fmt: skip
    for k, (model, key) in enumerate((("pu_pre_dft", "pre"), ("pu_post_dft", "post"))):
        d = summary[summary["baseline"] == model].set_index("scheme").reindex(SCHEMES)
        xs = x + (k - 0.5) * 0.22
        err = np.vstack([d["tpr_mean"] - d["tpr_ci95_low"], d["tpr_ci95_high"] - d["tpr_mean"]])
        ax.errorbar(xs, d["tpr_mean"], yerr=err, fmt="o", ms=4.5, color=SERIES[k], lw=1.2,
                    capsize=0, mec=SURFACE_RING, mew=0.8, label=t[key], zorder=3)  # fmt: skip
    ax.set_xticks(x, [SCHEME_LABEL[lang][s] for s in SCHEMES])
    ax.set_ylabel(t["y"])
    ax.set_ylim(0.6, 1.0)
    ax.set_xlim(-0.6, len(SCHEMES) - 0.4)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left", ncols=2, handletextpad=0.3, columnspacing=1.2)
    return fig


SURFACE_RING = "#ffffff"  # anillo del color del fondo alrededor de cada marcador


def main() -> None:
    setup_style()
    summary = pd.read_csv(SRC)
    for lang in TEXT:
        save(draw(summary, lang), NAME, lang)
    write_provenance(FIG_DIR / f"{NAME}.json", [SRC], "TPR media e IC 95 % de 10 semillas.")


if __name__ == "__main__":
    main()
