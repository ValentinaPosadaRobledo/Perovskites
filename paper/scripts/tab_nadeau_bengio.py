"""Tabla: comparaciones secundarias del PU `pre_dft` con las líneas base (Nadeau–Bengio, F4.2).

Entrada: `results/f4_2_stats/nadeau_bengio.csv` (`scripts/report_f4_2.py`; preregistro §9).
TPR a P(ŷ = 1) = 0,10 fold a fold sobre las composiciones puntuadas por los dos modelos; Δ =
TPR_PU − TPR_base con su IC 95 % del t corregido; J = semillas × folds con positivos; Holm
sobre las 6 comparaciones de cada esquema. No confirmatorio.

Salidas en `paper/tables/`: `tab_nadeau_bengio.{md,tex}` (inglés) y `_es`, con su `.json`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, SCHEME_LABEL
from tab_prior_and_formability import write

SRC = ROOT / "results" / "f4_2_stats" / "nadeau_bengio.csv"
DEC = {"en": ".", "es": ","}
BASELINE = {
    "pu_post_dft": ("PU post-DFT", "PU post-DFT"),
    "ehull_model": ("E_hull only", "Solo E_hull"),
    "symmetry_model": ("Symmetry only", "Solo simetría"),
    "tau_bartel": ("τ < 4.18", "τ < 4,18"),
    "t_window_bartel": ("t window (Bartel)", "Ventana de t (Bartel)"),
    "charge_balance_icsd": ("Charge balance (ICSD)", "Balance de carga (ICSD)"),
}
HEAD = {
    "en": {"scheme": "Split", "base": "Baseline", "tpr_pu": "TPR PU", "tpr_b": "TPR baseline",
           "d": "ΔTPR [95 % CI]", "j": "J", "p": "p (Holm)"},
    "es": {"scheme": "Partición", "base": "Línea base", "tpr_pu": "TPR PU",
           "tpr_b": "TPR línea base", "d": "ΔTPR [IC 95 %]", "j": "J", "p": "p (Holm)"},
}  # fmt: skip


def num(v: float, lang: str, signed: bool = False) -> str:
    """Tres decimales, con signo si se pide; coma decimal en español."""
    s = f"{v:+.3f}" if signed else f"{v:.3f}"
    return s.replace(".", DEC[lang])


def pval(p: float, lang: str) -> str:
    """Valor p analítico: tres decimales, o < 0,001."""
    s = "< 0.001" if p < 1e-3 else f"{p:.3f}"
    return s.replace(".", DEC[lang])


def main() -> None:
    nb = pd.read_csv(SRC)
    for lang in ("en", "es"):
        h, k, sfx = HEAD[lang], (0 if lang == "en" else 1), ("" if lang == "en" else "_es")
        sep = ", " if lang == "en" else "; "
        rows = [{h["scheme"]: SCHEME_LABEL[lang][r.scheme],
                 h["base"]: BASELINE[r.comparator][k],
                 h["tpr_pu"]: num(r.tpr_ref_fold_mean, lang),
                 h["tpr_b"]: num(r.tpr_comp_fold_mean, lang),
                 h["d"]: f"{num(r['mean'], lang, True)} [{num(r.ci_low, lang, True)}{sep}"
                         f"{num(r.ci_high, lang, True)}]",
                 h["j"]: int(r.n_diffs),
                 h["p"]: pval(r.p_holm, lang) + (" *" if r.significant else "")}
                for _, r in nb.iterrows()]  # fmt: skip
        cap = ("Secondary comparisons (not confirmatory): PU pre-DFT vs. each baseline, TPR at "
               "P(ŷ = 1) = 0.10 per fold on the compositions scored by both; corrected resampled "
               "t-test (Nadeau and Bengio), J = seeds × folds with positives; Holm within each "
               "split; * p < 0.05. TPR is the mean over folds." if lang == "en" else
               "Comparaciones secundarias (no confirmatorias): PU pre-DFT frente a cada línea "
               "base, TPR a P(ŷ = 1) = 0,10 en cada fold sobre las composiciones puntuadas por "
               "las dos; t corregido de Nadeau y Bengio, J = semillas × folds con positivos; "
               "Holm dentro de cada partición; * p < 0,05. La TPR es la media por "
               "fold.")  # fmt: skip
        write(pd.DataFrame(rows), f"tab_nadeau_bengio{sfx}", cap, [SRC])


if __name__ == "__main__":
    main()
