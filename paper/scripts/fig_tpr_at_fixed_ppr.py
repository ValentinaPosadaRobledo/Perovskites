"""Figura: TPR a P(ŷ = 1) fija, PU `pre_dft` frente a τ en las MISMAS composiciones.

Entrada: `results/f3_3_pu/summary_same_subset.csv`, bloque "composiciones con τ definido": el
PU y τ se ordenan por su puntuación y se umbralizan a la misma P(ŷ = 1) en cada fold (§7.2–7.3).
Así se compara solo la ordenación, sin el umbral externo de τ. Un panel por P(ŷ = 1) fija;
media e IC 95 % de 10 semillas (τ es determinista, pero sus folds de `random` y `cluster`
cambian con la semilla).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    FIG_DIR,
    ROOT,
    SCHEME_LABEL,
    SERIES,
    plt,
    save,
    setup_style,
    write_provenance,
)

NAME = "fig_tpr_at_fixed_ppr"
SRC = ROOT / "results" / "f3_3_pu" / "summary_same_subset.csv"
ORDER = ["random", "loeo_A", "loeo_B", "family", "cluster"]
PPRS = ["0.05", "0.10", "0.20"]
MODELS = [("pu_pre_dft", "pu"), ("tau_bartel", "tau")]
TEXT = {
    "en": {"y": "TPR at fixed P(ŷ = 1)", "panel": "P(ŷ = 1) = {p}", "pu": "PU pre-DFT",
           "tau": "τ (Bartel et al.)"},
    "es": {"y": "TPR a P(ŷ = 1) fija", "panel": "P(ŷ = 1) = {p}", "pu": "PU pre-DFT",
           "tau": "τ (Bartel et al.)"},
}  # fmt: skip


def draw(same: pd.DataFrame, lang: str) -> plt.Figure:
    """Tres paneles (una P(ŷ = 1) cada uno), mismo eje y; dos series por panel."""
    t = TEXT[lang]
    block = same[same["subset"] == "tau_defined"]
    fig, axes = plt.subplots(1, len(PPRS), figsize=(7.0, 2.6), sharey=True)
    x = np.arange(len(ORDER))
    for ax, ppr in zip(axes, PPRS, strict=True):
        col = f"tpr_at_ppr_{ppr}"
        for k, (model, key) in enumerate(MODELS):
            d = block[block["baseline"] == model].set_index("scheme").reindex(ORDER)
            err = np.vstack([d[f"{col}_mean"] - d[f"{col}_ci95_low"],
                             d[f"{col}_ci95_high"] - d[f"{col}_mean"]])  # fmt: skip
            ax.errorbar(x + (k - 0.5) * 0.24, d[f"{col}_mean"], yerr=err, fmt="o", ms=4,
                        color=SERIES[k], lw=1.1, capsize=0, mec="#ffffff", mew=0.7,
                        label=t[key], zorder=3)  # fmt: skip
        ax.set_title(t["panel"].format(p=ppr.replace(".", "," if lang == "es" else ".")),
                     loc="left")  # fmt: skip
        ax.set_xticks(x, [SCHEME_LABEL[lang][s] for s in ORDER], rotation=35, ha="right")
        ax.set_xlim(-0.6, len(ORDER) - 0.4)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel(t["y"])
    axes[0].set_ylim(0, None)
    axes[0].legend(loc="upper left", handletextpad=0.3)
    fig.tight_layout(w_pad=1.0)
    return fig


def main() -> None:
    setup_style()
    same = pd.read_csv(SRC)
    for lang in TEXT:
        save(draw(same, lang), NAME, lang)
    write_provenance(FIG_DIR / f"{NAME}.json", [SRC],
                     "Composiciones con τ definido; media e IC 95 % de 10 semillas.")  # fmt: skip


if __name__ == "__main__":
    main()
