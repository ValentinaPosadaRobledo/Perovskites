"""Informe de F1.4: concordancia entre la identificación automática y la revisión manual.

Lee `docs/validacion_manual_casos.csv` (veredictos de la revisora) y la tabla de F1.3, y
escribe en `results/f1_4_validacion_manual/`:
    summary.json   cifras del informe
    informe.md     informe legible, generado (no editar a mano)

Uso:
    uv run python scripts/report_f1_4.py
"""

from __future__ import annotations

import json

import pandas as pd
import yaml
from scipy.stats import beta

from perovskite_pu.utils.paths import CONFIGS_DIR, INTERIM_DIR, RAW_DIR, RESULTS_DIR, ROOT
from perovskite_pu.utils.run_metadata import git_commit, git_is_dirty

OUT = RESULTS_DIR / "f1_4_validacion_manual"
# Equivalencia entre las clases automáticas y las etiquetas de la revisora.
MANUAL_TO_AUTO = {
    "perovskita": "perovskite",
    "no_perovskita": "non_perovskite",
    "politipo_hexagonal": "hexagonal_polytype",
    "polianion": "polyanion",
    "dudosa": "doubtful",
}
DECISIVE = ["perovskite", "non_perovskite", "hexagonal_polytype", "polyanion"]


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Intervalo exacto (Clopper–Pearson) para una proporción k/n."""
    a = (1 - level) / 2
    lo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))
    return lo, hi


def composition_labels(t: pd.DataFrame, cls_col: str) -> pd.Series:
    """Etiqueta por composición (vista previa de F1.5) a partir de la clase final por entrada."""

    def lab(g: pd.DataFrame) -> str:
        if (g[cls_col] == "polyanion").all():
            return "polyanion"
        e = g[g["experimental"]]
        if e.empty:
            return "unlabeled"
        if (e[cls_col] == "perovskite").any():
            return "positive"
        if e[cls_col].isin(["doubtful", "perovskite_borderline"]).any():
            return "ambiguous"
        return "synthesized_non_perovskite"

    return t.groupby("reduced_formula").apply(lab, include_groups=False)


def main() -> None:
    cfg = yaml.safe_load((CONFIGS_DIR / "data.yaml").read_text("utf-8"))
    d = pd.read_csv(ROOT / cfg["labeling"]["manual_review"], encoding="utf-8-sig")
    d["manual"] = d["clasificacion_manual"].map(MANUAL_TO_AUTO)
    if d["manual"].isna().any():
        raise ValueError(
            f"Etiquetas manuales no reconocidas: {sorted(set(d.loc[d['manual'].isna(), 'clasificacion_manual']))}"
        )

    decisive = d[d["clasificacion_automatica"].isin(DECISIVE)]
    k, n = int((decisive["manual"] == decisive["clasificacion_automatica"]).sum()), len(decisive)
    lo, hi = clopper_pearson(k, n)
    border = d[d["clasificacion_automatica"] == "perovskite_borderline"]
    discord = d[
        (d["clasificacion_automatica"].isin(DECISIVE))
        & (d["manual"] != d["clasificacion_automatica"])
    ]

    # Efecto sobre las etiquetas por composición (la revisión prevalece en las entradas revisadas).
    t = pd.read_parquet(INTERIM_DIR / "f1_3_identification.parquet").merge(
        pd.read_parquet(RAW_DIR / "mp_abo3_index.parquet"), on="material_id"
    )
    t["experimental"] = ~t["theoretical"].astype(bool)
    rev = d.set_index("material_id")["manual"]
    t["cls_final"] = [rev.get(m, c) for m, c in zip(t["material_id"], t["cls"], strict=True)]
    before, after = composition_labels(t, "cls"), composition_labels(t, "cls_final")
    p, u, amb = (
        (after == "positive").sum(),
        (after == "unlabeled").sum(),
        (after == "ambiguous").sum(),
    )

    s = {
        "code_commit": git_commit(), "code_dirty": git_is_dirty(),
        "n_cases": len(d), "reviewer": sorted(d["revisora"].dropna().unique()), "dates": sorted(d["fecha"].dropna().astype(str).unique()),
        "decisive_agreement": {"k": k, "n": n, "rate": k / n, "ci95_clopper_pearson": [lo, hi]},
        "confusion": pd.crosstab(d["clasificacion_automatica"], d["clasificacion_manual"]).to_dict(),
        "borderline_outcomes": border["clasificacion_manual"].value_counts().to_dict(),
        "decisive_discordant": discord[["caso", "formula_ABO3", "material_id"]].to_dict("records"),
        "composition_labels_before": before.value_counts().to_dict(),
        "composition_labels_after": after.value_counts().to_dict(),
        "pu_set": int(p + u), "labeled_prevalence": float(p / (p + u)),
        "labeled_prevalence_if_ambiguous_positive": float((p + amb) / (p + amb + u)),
        "ambiguous_compositions": sorted(after[after == "ambiguous"].index),
    }  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(s, indent=2, ensure_ascii=False) + "\n", "utf-8")

    conf = pd.crosstab(
        d["clasificacion_automatica"], d["clasificacion_manual"], margins=True, margins_name="Total"
    )
    rows_b = "\n".join(
        f"| {r.caso} | {r.formula_ABO3} | {r.grupo_espacial} | `{r.mp_id_antiguo}` | {r.clasificacion_manual} | {r.motivo_referencia} |"
        for r in border.itertuples()
    )
    lab_tab = (
        pd.DataFrame(
            {"antes de la revisión": before.value_counts(), "después": after.value_counts()}
        )
        .fillna(0)
        .astype(int)
    )
    md = f"""# Informe F1.4: validación manual de la identificación de perovskitas

> Generado por `scripts/report_f1_4.py` (commit `{s["code_commit"][:7]}`{", árbol sucio" if s["code_dirty"] else ""}).
> No editar a mano. Revisora: {", ".join(s["reviewer"])} ({", ".join(s["dates"])}). Casos: {s["n_cases"]}.

## 1. Concordancia

{conf.to_markdown()}

**Clasificaciones firmes del algoritmo** (perovskita, no perovskita, politipo hexagonal,
polianión) **confirmadas por la revisora: {k}/{n} ({k / n:.0%}; IC 95 % Clopper–Pearson
{lo:.1%}–{hi:.1%}).** Esta es la validación independiente de la identificación: no comparte la
circularidad de la comparación con Bartel (F1.3).

## 2. Casos límite ({len(border)})

La categoría "límite" no se acepta como perovskita. La revisión confirma que funciona como
filtro: {s["borderline_outcomes"].get("perovskita", 0)} de {len(border)} resultaron perovskitas y el resto son no perovskitas
(tipo LiNbO₃, pirocloros) o dudosas.

| Caso | Fórmula | Grupo | ID MP | Veredicto | Motivo (revisora) |
|---|---|---|---|---|---|
{rows_b}

## 3. Hallazgos

1. **Tipo LiNbO₃ frente a perovskita:** la NC del sitio A por distancias no los separa
   (NaNbO₃ R3c tiene 12 O a < 3,3 Å y no es perovskita). Justifica no haber adoptado la regla
   de distancias para todos los A (F1.3).
2. **Falsos negativos recurrentes:** perovskitas Pnma en las que CrystalNN da NC 6 al sitio A
   (InRhO₃, LaYO₃, ScCrO₃). **No se ajusta el criterio** con estos 59 casos, para evitar el
   sobreajuste; en los casos revisados prevalece el veredicto manual (decisión 2026-11-14).
3. **Entradas de MP con roles A/B invertidos** (LaGaO₃, PrGaO₃, NdGaO₃, BaPbO₃, CdTiO₃ en
   Pm-3m): catión grande en el sitio octaédrico. No afectan a las etiquetas, porque esas
   composiciones tienen otras entradas experimentales perovskita correctas.
4. **Pirocloros (AgSbO₃, KSbO₃):** red de octaedros con vértices compartidos que no es
   perovskita; el algoritmo los dejó correctamente como límite por la NC(A) = 6.

## 4. Efecto sobre las etiquetas por composición (vista previa de F1.5)

{lab_tab.to_markdown()}

- Conjunto PU (P + U): **{s["pu_set"]}** composiciones. Prevalencia de positivas etiquetadas:
  **{s["labeled_prevalence"]:.1%}** ({s["labeled_prevalence_if_ambiguous_positive"]:.1%} si las ambiguas contaran como P).
- **Ambiguas ({amb})**, excluidas del análisis principal (P en la sensibilidad):
  {", ".join(s["ambiguous_compositions"])}.
"""
    (OUT / "informe.md").write_text(md, "utf-8")
    print(
        json.dumps(
            {
                k_: s[k_]
                for k_ in (
                    "decisive_agreement",
                    "borderline_outcomes",
                    "composition_labels_after",
                    "labeled_prevalence",
                )
            },
            indent=1,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":  # requisito en Windows (regla 2.2)
    main()
