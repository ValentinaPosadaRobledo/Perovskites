"""Tabla: features con más peso SHAP en el modelo PU `pre_dft` final y su estabilidad (F4.3).

Entradas (última corrida de `f4_3_shap` en `results/runs_index.csv`): `importance_by_seed.csv`
y `summary_features.csv`. Fracción media del |SHAP| total (10 semillas), rango de su puesto
entre semillas, frecuencia en el top 10 y SHAP medio con signo en P y en U (log-odds).

Salidas en `paper/tables/`: `tab_shap_features.{md,tex}` (inglés) y `_es`, con su `.json`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_shap_groups import label, run_dir
from tab_prior_and_formability import write

TOP = 10
DEC = {"en": ".", "es": ","}
HEAD = {
    "en": {"f": "Feature", "g": "Group", "share": "Share of importance", "rank": "Rank (range)",
           "top": "In top 10", "p": "Mean SHAP, P", "u": "Mean SHAP, U"},
    "es": {"f": "Feature", "g": "Grupo", "share": "Fracción de la importancia",
           "rank": "Puesto (rango)", "top": "En el top 10", "p": "SHAP medio, P",
           "u": "SHAP medio, U"},
}  # fmt: skip


def main() -> None:
    rd = run_dir()
    inputs = [rd / "importance_by_seed.csv", rd / "summary_features.csv"]
    by_seed, summary = (pd.read_csv(p) for p in inputs)
    share = by_seed.pivot(index="feature", columns="seed", values="share_all").mean(axis=1)
    top = summary.assign(share=summary["feature"].map(share)).nlargest(TOP, "share")
    n_seeds = by_seed["seed"].nunique()
    for lang in ("en", "es"):
        h, sfx = HEAD[lang], ("" if lang == "en" else "_es")
        num = lambda v, dec=DEC[lang]: f"{v:+.3f}".replace(".", dec)  # noqa: E731
        rows = [{h["f"]: r.feature, h["g"]: label(r.group, lang),
                 h["share"]: f"{r.share:.1%}".replace(".", DEC[lang]),
                 h["rank"]: f"{r.rank_mean:.1f} ({r.rank_min:.0f}–{r.rank_max:.0f})"
                            .replace(".", DEC[lang]),
                 h["top"]: f"{round(r.freq_top10 * n_seeds)}/{n_seeds}",
                 h["p"]: num(r.mean_signed_P), h["u"]: num(r.mean_signed_U)}
                for r in top.itertuples()]  # fmt: skip
        cap = ("Top features of the final PU pre-DFT model (10 seeds; path-dependent TreeSHAP "
               "on each bag's log-odds, averaged over bags). Rank range and top-10 frequency "
               "across seeds; signed SHAP > 0 raises the score." if lang == "en" else
               "Features con más peso del modelo PU pre-DFT final (10 semillas; TreeSHAP "
               "path-dependent sobre los log-odds de cada bolsa, media de las bolsas). Rango "
               "del puesto y frecuencia en el top 10 entre semillas; SHAP con signo > 0 sube "
               "la puntuación.")  # fmt: skip
        write(pd.DataFrame(rows), f"tab_shap_features{sfx}", cap, inputs)


if __name__ == "__main__":
    main()
