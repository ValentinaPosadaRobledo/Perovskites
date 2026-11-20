"""Tablas: prior de clase π (F3.2) y validación de formabilidad (F3.4).

Entradas: `results/f3_2_prior/summary.csv` y `results/f3_4_formability/summary.csv`.
Salidas en `paper/tables/`: `tab_prior.{md,tex}`, `tab_formability.{md,tex}` (inglés) y sus
versiones `_es`, cada una con su `.json` de procedencia. Media [IC 95 %] de 10 semillas; las
reglas son deterministas (un solo número).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, TAB_DIR, write_provenance

PRIOR = ROOT / "results" / "f3_2_prior" / "summary.csv"
FORM = ROOT / "results" / "f3_4_formability" / "summary.csv"
DEC = {"en": ".", "es": ","}
# (método, etiqueta en inglés, etiqueta en español, ¿principal?)
PRIOR_ROWS = [
    ("elkan_noto", "Elkan–Noto (RF, isotonic)", "Elkan–Noto (RF, isotónica)", True),
    ("elkan_noto_gb", "Elkan–Noto (GB, isotonic)", "Elkan–Noto (GB, isotónica)", False),
    ("elkan_noto_sigmoid", "Elkan–Noto (RF, sigmoid)", "Elkan–Noto (RF, sigmoide)", False),
    ("tice", "TIcE (range in 4 bins)", "TIcE (rango en 4 partes)", True),
    ("tice_quantile", "TIcE (quantile bins)", "TIcE (cuantiles)", False),
]
# (variante, modelo o regla, etiqueta en inglés, etiqueta en español)
FORM_ROWS = [
    ("family", "pu_pre_dft", "PU pre-DFT, B-family folds", "PU pre-DFT, folds de familia de B"),
    ("final", "pu_pre_dft", "PU pre-DFT, final model", "PU pre-DFT, modelo final"),
    ("rule", "tau_bartel", "τ < 4.18 (in-sample on TableS1)",
     "τ < 4,18 (in-sample en la TableS1)"),
    ("rule", "t_window_bartel", "t in [0.825, 1.059]", "t en [0,825; 1,059]"),
    ("rule", "charge_balance_icsd", "Charge balance (ICSD states)",
     "Balance de carga (estados de ICSD)"),
]  # fmt: skip
# Encabezados de columna por idioma.
HEAD = {
    "en": {"method": "Method", "hidden": "Hidden positives", "model": "Model or rule",
           "fpr": "FPR, synthesized non-perovskites", "fpr_poly": "FPR, polyanions",
           "auc": "ROC-AUC, P vs. negatives"},
    "es": {"method": "Método", "hidden": "Positivas ocultas", "model": "Modelo o regla",
           "fpr": "FPR, sintetizadas no perovskita", "fpr_poly": "FPR, polianiones",
           "auc": "ROC-AUC, P frente a negativos"},
}  # fmt: skip


def cell(row: pd.Series, col: str, lang: str) -> str:
    """'media [IC 95 %]', o un solo número si no varía entre semillas; '—' si no aplica."""
    m = row.get(f"{col}_mean")
    if m is None or pd.isna(m):
        return "—"
    lo, hi = row[f"{col}_ci95_low"], row[f"{col}_ci95_high"]
    fmt = lambda v: f"{v:.3f}".replace(".", DEC[lang])  # noqa: E731
    sep = "; " if lang == "es" else ", "
    return fmt(m) if abs(hi - lo) < 5e-4 else f"{fmt(m)} [{fmt(lo)}{sep}{fmt(hi)}]"


def latex_text(value: object) -> str:
    """Escapa el texto para LaTeX y pasa a modo matemático los símbolos griegos y subíndices."""
    s = str(value)
    for a, b in (("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("_", "\\_"),
                 ("#", "\\#"), ("*", "$^{*}$")):  # fmt: skip
        s = s.replace(a, b)
    for a, b in (("π\\_U", "$\\pi_U$"), ("τ", "$\\tau$"), ("–", "--"), ("Δ", "$\\Delta$"),
                 (" < ", " $<$ ")):  # fmt: skip
        s = s.replace(a, b)
    return s


def to_latex(table: pd.DataFrame) -> str:
    """Tabla booktabs sin dependencias (pandas.to_latex exigiría Jinja2)."""
    cols = "l" + "c" * (table.shape[1] - 1)
    lines = [f"\\begin{{tabular}}{{{cols}}}", "\\toprule",
             " & ".join(latex_text(c) for c in table.columns) + " \\\\", "\\midrule"]  # fmt: skip
    for row in table.itertuples(index=False):
        lines.append(" & ".join(latex_text(v) for v in row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def write(table: pd.DataFrame, name: str, caption: str, inputs: list[Path]) -> None:
    """Markdown y LaTeX (booktabs) de la misma tabla, con su procedencia."""
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    md = f"{caption}\n\n{table.to_markdown(index=False)}\n"
    (TAB_DIR / f"{name}.md").write_text(md, "utf-8")
    (TAB_DIR / f"{name}.tex").write_text(f"% {caption}\n{to_latex(table)}", "utf-8")
    write_provenance(TAB_DIR / f"{name}.json", inputs, caption)


def main() -> None:
    prior = pd.read_csv(PRIOR)
    form = pd.read_csv(FORM)
    for lang in ("en", "es"):
        h, sfx = HEAD[lang], ("" if lang == "en" else "_es")
        rows = []
        for method, en, es, principal in PRIOR_ROWS:
            r = prior[(prior["population"] == "all") & (prior["method"] == method)].iloc[0]
            label = (en if lang == "en" else es) + (" *" if principal else "")
            rows.append({h["method"]: label, "c": cell(r, "c", lang),
                         "π_U": cell(r, "pi_u", lang),
                         h["hidden"]: f"{r['hidden_positives_mean']:.0f}"})  # fmt: skip
        cap = ("Class prior on train_pool (193 P, 731 U); * principal methods. Mean [95 % CI] "
               "over 10 seeds; descriptive (SCAR does not hold)." if lang == "en" else
               "Prior de clase en train_pool (193 P, 731 U); * métodos principales. Media "
               "[IC 95 %] de 10 semillas; descriptiva (SCAR no se cumple).")  # fmt: skip
        write(pd.DataFrame(rows), f"tab_prior{sfx}", cap, [PRIOR])

        rows = []
        for variant, model, en, es in FORM_ROWS:
            r = form[(form["variant"] == variant) & (form["model"] == model)].iloc[0]
            rows.append({h["model"]: en if lang == "en" else es,
                         h["fpr"]: cell(r, "fpr_principal", lang),
                         h["fpr_poly"]: cell(r, "fpr_polyanion", lang),
                         h["auc"]: cell(r, "auc_principal", lang)})  # fmt: skip
        cap = ("Formability (not synthesizability): 73 synthesized non-perovskite ABO3 oxides as "
               "true negatives; 61 polyanions reported apart." if lang == "en" else
               "Formabilidad (no sintetizabilidad): 73 óxidos ABO3 sintetizados no perovskita "
               "como negativos verdaderos; 61 polianiones aparte.")  # fmt: skip
        write(pd.DataFrame(rows), f"tab_formability{sfx}", cap, [FORM])


if __name__ == "__main__":
    main()
