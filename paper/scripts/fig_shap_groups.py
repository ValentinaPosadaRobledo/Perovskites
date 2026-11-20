"""Figura: importancia SHAP por grupo de features del modelo PU `pre_dft` final (F4.3).

Entradas (última corrida de `f4_3_shap` en `results/runs_index.csv`):
`group_importance_by_seed.csv` y `summary_groups.csv`. TreeSHAP *path-dependent* sobre la
salida cruda de cada bolsa (log-odds: Gradient Boosting en las 10 semillas), media de las bolsas.

- (a) Fracción de la importancia total (media de |SHAP de grupo| sobre las 924 composiciones,
  normalizada por semilla): punto = media de 10 semillas; barra = mínimo–máximo entre semillas.
- (b) Dirección: SHAP medio con signo del grupo en las P y en las U (media de 10 semillas).

Se muestran los `TOP` grupos más importantes; el SHAP de un grupo es la suma con signo de sus
features.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import FIG_DIR, INK, ROOT, SERIES, plt, save, setup_style, write_provenance

NAME = "fig_shap_groups"
TOP = 12
LABEL = {
    "en": {"missing_t_mu": "t, μ undefined (no charge balance)", "t": "t (Goldschmidt)",
           "mu": "μ (octahedral)", "IonProperty": "Ionic properties",
           "ValenceOrbital": "Valence orbitals", "Stoichiometry": "Stoichiometry"},
    "es": {"missing_t_mu": "t, μ no definidos (sin balance de carga)", "t": "t (Goldschmidt)",
           "mu": "μ (octaédrico)", "IonProperty": "Propiedades iónicas",
           "ValenceOrbital": "Orbitales de valencia", "Stoichiometry": "Estequiometría"},
}  # fmt: skip
TEXT = {
    "en": {"a": "a  Share of total |SHAP|", "b": "b  Mean signed SHAP (log-odds)",
           "P": "positives (193)", "U": "unlabeled (731)", "magpie": "MAGPIE: "},
    "es": {"a": "a  Fracción del |SHAP| total", "b": "b  SHAP medio con signo (log-odds)",
           "P": "positivas (193)", "U": "no etiquetadas (731)", "magpie": "MAGPIE: "},
}  # fmt: skip
NAMED = {"missing_t_mu", "t", "mu", "IonProperty", "ValenceOrbital", "Stoichiometry"}


def run_dir() -> Path:
    """Carpeta de la última corrida de F4.3 registrada en `runs_index.csv`."""
    index = pd.read_csv(ROOT / "results" / "runs_index.csv")
    return ROOT / index[index["experiment"] == "f4_3_shap"].iloc[-1]["run_dir"]


def label(group: str, lang: str) -> str:
    """Etiqueta legible; las propiedades de MAGPIE llevan su nombre original."""
    return LABEL[lang].get(group, TEXT[lang]["magpie"] + group if group not in NAMED else group)


def draw(by_seed: pd.DataFrame, summary: pd.DataFrame, lang: str) -> plt.Figure:
    """Dos paneles con los mismos grupos, ordenados por importancia media."""
    t = TEXT[lang]
    share = by_seed.pivot(index="group", columns="seed", values="share_all")
    order = share.mean(axis=1).sort_values(ascending=False).index[:TOP]
    y = np.arange(len(order))[::-1]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True,
                                   gridspec_kw={"width_ratios": [1.1, 1]})  # fmt: skip

    # (a) Fracción del total: media y rango entre semillas.
    m, lo, hi = share.mean(axis=1)[order], share.min(axis=1)[order], share.max(axis=1)[order]
    ax1.hlines(y, lo, hi, color=SERIES[0], lw=1.4, zorder=2)
    ax1.plot(m, y, "o", ms=5, color=SERIES[0], mec="#ffffff", mew=0.8, zorder=3)
    ax1.set_title(t["a"], loc="left")
    ax1.set_xlim(0, float(hi.max()) * 1.12)
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax1.set_yticks(y, [label(g, lang) for g in order])
    ax1.tick_params(axis="y", length=0)
    ax1.grid(axis="y", visible=False)

    # (b) Dirección del efecto en P y en U.
    s = summary.set_index("group").loc[order]
    ax2.axvline(0, color=INK["secondary"], lw=0.7, zorder=1)
    for k, (col, key) in enumerate((("mean_signed_P", "P"), ("mean_signed_U", "U"))):
        ax2.plot(s[col], y + (0.5 - k) * 0.3, "o", ms=5, color=SERIES[k], mec="#ffffff",
                 mew=0.8, zorder=3, label=t[key])  # fmt: skip
    ax2.set_title(t["b"], loc="left")
    ax2.grid(axis="y", visible=False)
    ax2.legend(loc="upper center", handletextpad=0.3)  # hueco entre P y U arriba
    fig.tight_layout(w_pad=1.0)
    return fig


def main() -> None:
    setup_style()
    rd = run_dir()
    inputs = [rd / "group_importance_by_seed.csv", rd / "summary_groups.csv"]
    by_seed, summary = (pd.read_csv(p) for p in inputs)
    for lang in TEXT:
        save(draw(by_seed, summary, lang), NAME, lang)
    note = "Media y mínimo–máximo de 10 semillas; TreeSHAP path-dependent, media de las bolsas."
    write_provenance(FIG_DIR / f"{NAME}.json", inputs, note)


if __name__ == "__main__":
    main()
