"""Informe de F1.6: conteos por etiqueta, estructura, elemento A, elemento B y familia.

Lee las salidas de F1.5 (`data/interim/f1_5_labels.parquet`, `f1_5_entries.parquet`) y
escribe en `results/f1_6_conteos/`:
    counts_by_A.csv, counts_by_B.csv, counts_by_family.csv
    summary.json, informe.md (generado; no editar a mano)

Incluye el punto de decisión del ROADMAP (≥ ~100 positivas ABO3 → se mantiene el alcance) y
un avance para F1.7 (elementos con ≥ 5 positivas para loeo_A/loeo_B).

Uso:
    uv run python scripts/report_f1_6.py
"""

from __future__ import annotations

import json

import pandas as pd
import yaml

from perovskite_pu.splits.schemes import assign_b_family
from perovskite_pu.utils.paths import CONFIGS_DIR, INTERIM_DIR, RESULTS_DIR
from perovskite_pu.utils.run_metadata import git_commit, git_is_dirty

OUT = RESULTS_DIR / "f1_6_conteos"
COLS = ["positive", "unlabeled", "synthesized_non_perovskite", "ambiguous"]
SHORT = {
    "positive": "P",
    "unlabeled": "U",
    "synthesized_non_perovskite": "sint. no perov.",
    "ambiguous": "ambigua",
}


def by_site(labels: pd.DataFrame, site: str) -> pd.DataFrame:
    """Composiciones por elemento del sitio dado y etiqueta (sin polianiones)."""
    t = pd.crosstab(labels[site], labels["label"]).reindex(columns=COLS, fill_value=0)
    t["total_PU"] = t["positive"] + t["unlabeled"]
    return t.sort_values(["positive", "total_PU"], ascending=False)


def main() -> None:
    cfg = yaml.safe_load((CONFIGS_DIR / "data.yaml").read_text("utf-8"))
    L = pd.read_parquet(INTERIM_DIR / "f1_5_labels.parquet")
    E = pd.read_parquet(INTERIM_DIR / "f1_5_entries.parquet")
    lab = L[L["label"] != "polyanion"].copy()
    lab["family"] = assign_b_family(lab["B"], cfg["b_families"])
    min_pos = cfg["splits"]["loeo_B"]["min_positives_per_element"]

    counts = L["label"].value_counts()
    p, u = int(counts.get("positive", 0)), int(counts.get("unlabeled", 0))
    ub = int(((L["label"] == "unlabeled") & ~L["flag_no_charge_balance"]).sum())
    A, B = by_site(lab, "A"), by_site(lab, "B")
    fam = pd.crosstab(lab["family"], lab["label"]).reindex(columns=COLS, fill_value=0)
    fam["total_PU"] = fam["positive"] + fam["unlabeled"]

    # Estructura: sistema cristalino y grupo espacial de las entradas perovskita experimentales.
    pe = E[(E["cls_final"] == "perovskite") & E["experimental"]]
    cs = pe["crystal_system"].value_counts()
    sg = pe["spacegroup_symbol"].value_counts().head(10)
    by_cls = pd.crosstab(
        E["cls_final"], E["experimental"].map({True: "experimental", False: "teórica"})
    )

    s = {
        "code_commit": git_commit(), "code_dirty": git_is_dirty(),
        "labels": counts.to_dict(), "entries": len(E),
        "decision_point": {"positives": p, "threshold": 100, "decision": "mantener alcance ABO3 óxidos" if p >= 100 else "ampliar"},
        "pu_set": p + u, "labeled_prevalence": p / (p + u),
        "pu_set_charge_balanced_U": p + ub, "labeled_prevalence_charge_balanced_U": p / (p + ub),
        "n_A_elements_PU": int((A["total_PU"] > 0).sum()), "n_B_elements_PU": int((B["total_PU"] > 0).sum()),
        "A_elements_with_ge_min_positives": int((A["positive"] >= min_pos).sum()),
        "B_elements_with_ge_min_positives": int((B["positive"] >= min_pos).sum()),
        "min_positives_per_element": min_pos,
        "families": fam.to_dict("index"),
        "perovskite_experimental_crystal_systems": cs.to_dict(),
    }  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(s, indent=2, ensure_ascii=False) + "\n", "utf-8")
    for name, t in (("A", A), ("B", B), ("family", fam)):
        t.to_csv(OUT / f"counts_by_{name}.csv", lineterminator="\n")

    def top(t: pd.DataFrame, n: int = 15) -> str:
        return t.head(n).rename(columns=SHORT).to_markdown()

    small_b = B[(B["positive"] > 0) & (B["positive"] < min_pos)]
    md = (
        f"""# Informe F1.6: conteos

> Generado por `scripts/report_f1_6.py` (commit `{s["code_commit"][:7]}`{", árbol sucio" if s["code_dirty"] else ""}).
> No editar a mano. Tablas completas: `counts_by_A.csv`, `counts_by_B.csv`, `counts_by_family.csv`.

## 1. Punto de decisión del ROADMAP

**{p} composiciones ABO₃ positivas** (umbral: ~100; estimación previa: 200–350). **Decisión:
{s["decision_point"]["decision"]}**, sin ampliar a otras clases de perovskita.

## 2. Etiquetas

| Etiqueta | Composiciones |
|---|---|
"""
        + "\n".join(f"| {k} | {v} |" for k, v in counts.items())
        + f"""

- **Conjunto PU: {p + u}**; prevalencia de positivas etiquetadas **{p / (p + u):.1%}**.
- Sensibilidad, solo con U que admiten balance de carga: PU = {p + ub}; prevalencia {p / (p + ub):.1%}.

## 3. Estructura de las perovskitas experimentales ({len(pe)} entradas)

**Sistema cristalino:** {", ".join(f"{k} {v}" for k, v in cs.items())}.

**Grupos espaciales más frecuentes:** {", ".join(f"{k} ({v})" for k, v in sg.items())}.

**Todas las entradas por clase final:**

{by_cls.to_markdown()}

## 4. Por familia del elemento B

{fam.rename(columns=SHORT).to_markdown()}

## 5. Por elemento B (15 con más positivas; {s["n_B_elements_PU"]} elementos B en P ∪ U)

{top(B)}

## 6. Por elemento A (15 con más positivas; {s["n_A_elements_PU"]} elementos A en P ∪ U)

{top(A)}

## 7. Avance para F1.7

- Elementos **B** con ≥ {min_pos} positivas: **{s["B_elements_with_ge_min_positives"]}**. Con menos, se fusionan por familia
  (`small_group_policy: merge_family`): {len(small_b)} elementos B con entre 1 y {min_pos - 1} positivas.
- Elementos **A** con ≥ {min_pos} positivas: **{s["A_elements_with_ge_min_positives"]}** (los demás se reportan aparte,
  `report_separately`).
- `family` tiene {len(fam)} grupos; el holdout (F1.8) reserva elementos B estratificando por
  familia hasta cubrir ~10–15 % de las positivas.
"""
    )
    (OUT / "informe.md").write_text(md, "utf-8")
    print(
        json.dumps(
            {
                k: s[k]
                for k in (
                    "decision_point",
                    "pu_set",
                    "B_elements_with_ge_min_positives",
                    "A_elements_with_ge_min_positives",
                )
            },
            indent=1,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":  # requisito en Windows (regla 2.2)
    main()
