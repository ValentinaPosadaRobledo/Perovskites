"""Figura: pruebas confirmatorias de H1 y H2 sobre la validación cruzada (F4.2).

Entradas: `results/f4_2_stats/h1.csv`, `h2_principal.csv` y `h2_sin_bartel2019.csv` (bootstrap
jerárquico pareado por composición, B = 10 000, Holm sobre los 4 esquemas; preregistro §8–9).

- (a) H1: caída de TPR del PU `pre_dft` de `random` a cada esquema, con su IC 95 %; línea
  discontinua en δ₁ = 0,05. Marcador lleno si el esquema cumple el criterio (p de Holm < 0,05 y
  caída ≥ δ₁); vacío si no.
- (b) H2: Δ = TPR_PU − TPR_τ a la P(ŷ = 1) de τ, con su IC 95 %; línea discontinua en
  −δ₂ = −0,05. Dos series: análisis principal (τ *in-sample* en los compuestos de Bartel) y sin
  ellos. Marcador lleno si la no inferioridad se demuestra (p unilateral de Holm < 0,025).

El relleno es la codificación secundaria del resultado; la identidad de la serie la dan el
color y la leyenda.
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

NAME = "fig_h1_h2_confirmatory"
SRC = ROOT / "results" / "f4_2_stats"
H1 = SRC / "h1.csv"
H2 = {"principal": SRC / "h2_principal.csv", "sin_bartel2019": SRC / "h2_sin_bartel2019.csv"}
DELTA1 = DELTA2 = 0.05  # preregistro §8
TEXT = {
    "en": {"a": "a  H1: TPR drop from random split", "b": "b  H2: PU − τ at τ's P(ŷ = 1)",
           "ya": "ΔTPR (random − scheme)", "yb": "ΔTPR (PU − τ)",
           "d1": "δ₁", "d2": "−δ₂",
           "principal": "with Bartel's compounds (193 P)",
           "sin_bartel2019": "without them (41 P)",
           "met": "filled: criterion met (Holm)", "p": "p_Holm"},
    "es": {"a": "a  H1: caída de TPR desde la partición aleatoria",
           "b": "b  H2: PU − τ a la P(ŷ = 1) de τ",
           "ya": "ΔTPR (aleatorio − esquema)", "yb": "ΔTPR (PU − τ)",
           "d1": "δ₁", "d2": "−δ₂",
           "principal": "con los compuestos de Bartel (193 P)",
           "sin_bartel2019": "sin ellos (41 P)",
           "met": "lleno: cumple el criterio (Holm)", "p": "p_Holm"},
}  # fmt: skip


def fmt_p(p: float, lang: str) -> str:
    """Valor p del bootstrap: < 1e-4 si es 0 (§9); coma decimal en español."""
    s = "<1e-4" if p == 0 else f"{p:.3f}"
    return s.replace(".", ",") if lang == "es" else s


def point(ax, x: float, row: pd.Series, met: bool, color: str, label: str | None) -> None:
    """Estimación con IC 95 %; relleno según se cumpla o no el criterio."""
    ax.errorbar(x, row["estimate"], yerr=[[row["estimate"] - row["ci_low"]],
                [row["ci_high"] - row["estimate"]]], fmt="o", ms=5, color=color, lw=1.2,
                capsize=0, mfc=color if met else "#ffffff", mec=color, mew=1.2, zorder=3,
                label=label)  # fmt: skip


def draw(h1: pd.DataFrame, h2: dict[str, pd.DataFrame], lang: str) -> plt.Figure:
    """Dos paneles: H1 (una serie) y H2 (análisis principal y sin Bartel)."""
    t = TEXT[lang]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9))
    x = np.arange(len(CONFIRMATORY))
    labels = [SCHEME_LABEL[lang][s] for s in CONFIRMATORY]

    # (a) H1: caída con su IC y el umbral δ₁.
    ax1.axhline(0, color=INK["secondary"], lw=0.7, zorder=1)
    ax1.axhline(DELTA1, color=INK["muted"], lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax1.text(len(x) - 0.5, DELTA1, t["d1"], ha="right", va="bottom", color=INK["secondary"],
             fontsize=7)  # fmt: skip
    for i, s in enumerate(CONFIRMATORY):
        r = h1.loc[s]
        point(ax1, i, r, bool(r["criterion"]), SERIES[0], None)
        ax1.text(i, r["ci_high"] + 0.012, f"{t['p']} {fmt_p(r['p_holm'], lang)}", ha="center",
                 va="bottom", color=INK["secondary"], fontsize=6.5)  # fmt: skip
    ax1.set_title(t["a"], loc="left")
    ax1.set_ylabel(t["ya"])
    ax1.set_ylim(-0.05, 0.32)

    # (b) H2: Δ con su IC y el margen −δ₂, para las dos variantes.
    ax2.axhline(0, color=INK["secondary"], lw=0.7, zorder=1)
    ax2.axhline(-DELTA2, color=INK["muted"], lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax2.text(len(x) - 0.5, -DELTA2, t["d2"], ha="right", va="top", color=INK["secondary"],
             fontsize=7)  # fmt: skip
    for k, (analysis, tab) in enumerate(h2.items()):
        for i, s in enumerate(CONFIRMATORY):
            r = tab.loc[s]
            point(ax2, i + (k - 0.5) * 0.24, r, bool(r["noninferior"]), SERIES[k],
                  t[analysis] if i == 0 else None)  # fmt: skip
    ax2.set_title(t["b"], loc="left")
    ax2.set_ylabel(t["yb"])
    ax2.set_ylim(-0.37, 0.15)
    ax2.legend(loc="lower left", handletextpad=0.3)

    for ax in (ax1, ax2):
        ax.set_xticks(x, labels)
        ax.set_xlim(-0.5, len(x) - 0.5)
        ax.grid(axis="x", visible=False)
    fig.text(0.99, -0.02, t["met"], ha="right", va="top", color=INK["secondary"], fontsize=7)
    fig.tight_layout(w_pad=2.0)
    return fig


def main() -> None:
    setup_style()
    h1 = pd.read_csv(H1, index_col="scheme")
    h2 = {k: pd.read_csv(p, index_col="scheme") for k, p in H2.items()}
    for lang in TEXT:
        save(draw(h1, h2, lang), NAME, lang)
    write_provenance(FIG_DIR / f"{NAME}.json", [H1, *H2.values()],
                     "Estimación e IC 95 % del bootstrap jerárquico (B = 10 000); Holm sobre "
                     "los 4 esquemas confirmatorios.")  # fmt: skip


if __name__ == "__main__":
    main()
