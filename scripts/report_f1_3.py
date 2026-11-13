"""Informe de F1.3: identificación de perovskitas por conectividad.

Lee `data/interim/f1_3_identification.parquet` (generado por
`uv run python -m perovskite_pu.data.identify --n-jobs 4`) y escribe en
`results/f1_3_identificacion/`:
    summary.json     cifras del informe
    borderline.csv   casos límite para la validación manual (F1.4)
    informe.md       informe legible, generado (no editar a mano)

Uso:
    uv run python scripts/report_f1_3.py
"""

from __future__ import annotations

import json

import pandas as pd
import yaml

from perovskite_pu.data.external import (
    BARTEL_DIR,
    gu_root,
    label_conflicts,
    load_bartel2019,
    load_gu2022,
    to_legacy_mpid,
)
from perovskite_pu.utils.hashing import sha256_file
from perovskite_pu.utils.paths import CONFIGS_DIR, INTERIM_DIR, RAW_DIR, RESULTS_DIR
from perovskite_pu.utils.run_metadata import git_commit, git_is_dirty

OUT = RESULTS_DIR / "f1_3_identificacion"
TABLE = INTERIM_DIR / "f1_3_identification.parquet"


def sensitivity_strict(t: pd.DataFrame, angle: float, a_min: int, a_max: int) -> pd.Series:
    """Reclasifica con la regla original: ángulo ≥ `angle` y NC(A) CrystalNN en [a_min, a_max].

    Solo afecta a las entradas con red de octaedros que comparten vértices (clases
    perovskite y perovskite_borderline); el resto no cambia.
    """
    corner = t["cls"].isin(["perovskite", "perovskite_borderline"])
    ok = (
        corner
        & (t["bob_angle_min"] >= angle)
        & t["a_cn_min"].between(a_min, a_max)
        & t["a_cn_max"].between(a_min, a_max)
    )
    return t["cls"].where(~corner, ok.map({True: "perovskite", False: "perovskite_borderline"}))


def bartel_validation(exp: pd.DataFrame, col: str, bartel: pd.Series) -> dict:
    """Recall y falsos positivos frente a Bartel, por composición (entradas experimentales)."""
    comp = exp.groupby("reduced_formula")[col].apply(lambda c: (c == "perovskite").any())
    j = bartel.index.intersection(comp.index)
    tp = int((bartel[j] & comp[j]).sum())
    fn = int((bartel[j] & ~comp[j]).sum())
    fp = sorted(c for c in j if not bartel[c] and comp[c])
    neg = int((~bartel[j]).sum())
    return {"n": len(j), "recall": tp / (tp + fn), "tp": tp, "fn": fn,
            "fp": len(fp), "fp_rate": len(fp) / neg, "fp_list": fp,
            "missed": sorted(c for c in j if bartel[c] and not comp[c]),
            "positive_compositions": int(comp.sum())}  # fmt: skip


def main() -> None:
    cfg = yaml.safe_load((CONFIGS_DIR / "data.yaml").read_text("utf-8"))
    idc = cfg["identification"]
    idx = pd.read_parquet(RAW_DIR / "mp_abo3_index.parquet")
    t = pd.read_parquet(TABLE).merge(idx, on="material_id")
    t["legacy_id"] = t["material_id"].map(to_legacy_mpid)
    t["experimental"] = ~t["theoretical"].astype(bool)
    oc = idc["octahedral_connectivity"]
    t["cls_sens_strict140"] = sensitivity_strict(t, oc["sensitivity_min_b_o_b_angle_deg"],
                                                 idc["coordination"]["a_site_cn_min"],
                                                 idc["coordination"]["a_site_cn_max"])  # fmt: skip
    exp = t[t["experimental"]]

    bartel_df = load_bartel2019(BARTEL_DIR / cfg["external_sources"]["bartel2019"]["table"])
    bartel = bartel_df[bartel_df["abo3_oxide"]].set_index("reduced_formula")["is_perovskite"]
    gu = load_gu2022(gu_root(), cfg["external_sources"]["gu2022"]["split_index"])
    conflicts = label_conflicts(bartel_df, gu)

    def comp_class(c: str) -> str:
        e = exp[exp["reduced_formula"] == c]
        if e.empty:
            return "sin entrada experimental en MP"
        order = [
            "perovskite",
            "perovskite_borderline",
            "hexagonal_polytype",
            "non_perovskite",
            "polyanion",
        ]
        return next(k for k in order if (e["cls"] == k).any())

    conflicts["nuestra_clase"] = conflicts["reduced_formula"].map(comp_class)

    s = {
        "code_commit": git_commit(), "code_dirty": git_is_dirty(),
        "identification_table_sha256": sha256_file(TABLE),
        "criteria": {"min_b_o_b_angle_deg": oc["min_b_o_b_angle_deg"],
                     "a_site_cn": [idc["coordination"]["a_site_cn_min"], idc["coordination"]["a_site_cn_max"]],
                     "lone_pair_a_rule": idc["coordination"].get("lone_pair_a_rule")},
        "entries_by_class": t["cls"].value_counts().to_dict(),
        "experimental_entries_by_class": exp["cls"].value_counts().to_dict(),
        "oxidation": {"bv": int((t["oxi_method"] == "bv").sum()), "guess": int((t["oxi_method"] == "guess").sum()),
                      "none_non_polyanion": int((t["oxi_method"].isna() & (t["cls"] != "polyanion")).sum()),
                      "ambiguous": int(t["oxi_ambiguous"].sum())},
        "bartel_validation_main": bartel_validation(exp, "cls", bartel),
        "bartel_validation_sens_strict140": bartel_validation(exp, "cls_sens_strict140", bartel),
        "conflicts_bartel_gu": conflicts.groupby(["type", "nuestra_clase"]).size().reset_index(name="n").to_dict("records"),
    }  # fmt: skip
    proto = pd.crosstab(
        t["cls"],
        t["matches_cubic_prototype"].map({True: "coincide con prototipo", False: "no coincide"}),
    )
    s["prototype_vs_connectivity"] = proto.to_dict()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(s, indent=2, ensure_ascii=False) + "\n", "utf-8")
    cols = [
        "material_id",
        "legacy_id",
        "reduced_formula",
        "spacegroup_symbol",
        "experimental",
        "a_element",
        "b_element",
        "a_cn_min",
        "a_cn_max",
        "a_o_within_3p3",
        "bob_angle_min",
        "matches_cubic_prototype",
        "reasons",
    ]
    border = t[t["cls"] == "perovskite_borderline"].sort_values(
        ["experimental", "reduced_formula"], ascending=[False, True]
    )
    border[cols].to_csv(OUT / "borderline.csv", index=False, lineterminator="\n")

    m, st = s["bartel_validation_main"], s["bartel_validation_sens_strict140"]
    pv = proto.reindex(columns=["coincide con prototipo", "no coincide"], fill_value=0)
    perov_missed = int(pv.loc["perovskite", "no coincide"]) if "perovskite" in pv.index else 0
    nonperov_acc = int(
        pv.drop(index=["perovskite"], errors="ignore")["coincide con prototipo"].sum()
    )
    md = (
        f"""# Informe F1.3: identificación de perovskitas por conectividad

> Generado por `scripts/report_f1_3.py` (commit `{s["code_commit"][:7]}`{", árbol sucio" if s["code_dirty"] else ""}).
> No editar a mano. Cifras en `summary.json`; casos límite en `borderline.csv`.

## 1. Criterio aplicado (regla 2.1.1 de CLAUDE.md, decisiones del 2026-11-13)

Una entrada es **perovskita** si sus octaedros BO₆ comparten **solo vértices**, cada O une
exactamente 2 B, el ángulo B–O–B mínimo es ≥ **{oc["min_b_o_b_angle_deg"]:.0f}°** y la NC del sitio A
(CrystalNN) está entre {idc["coordination"]["a_site_cn_min"]} y {idc["coordination"]["a_site_cn_max"]}. Excepción para A con par libre
({", ".join(idc["coordination"]["lone_pair_a_rule"]["elements"])}): basta con ≥ {idc["coordination"]["lone_pair_a_rule"]["min_o"]} O a < {idc["coordination"]["lone_pair_a_rule"]["cutoff_ang"]} Å.
Sensibilidad: la regla original (140°, sin excepción).

## 2. Resultado

| Clase | Entradas | Entradas experimentales |
|---|---|---|
"""
        + "\n".join(
            f"| {k} | {v} | {s['experimental_entries_by_class'].get(k, 0)} |"
            for k, v in s["entries_by_class"].items()
        )
        + f"""

**Composiciones con al menos una entrada experimental perovskita: {m["positive_compositions"]}**
({st["positive_compositions"]} con la regla de sensibilidad). Es un avance de F1.5, que fija
formalmente las etiquetas.

**Estados de oxidación:** BVAnalyzer en {s["oxidation"]["bv"]} entradas, conjetura por composición en
{s["oxidation"]["guess"]}, ninguno en {s["oxidation"]["none_non_polyanion"]} (no polianiones; se usó CrystalNN sin
restricción catión–anión). Marcadas como ambiguas: {s["oxidation"]["ambiguous"]}.

## 3. Conectividad frente a similitud con el prototipo cúbico

{pv.to_markdown()}

- **{perov_missed} entradas perovskita no coinciden con el prototipo cúbico** (StructureMatcher, con
  las tolerancias por defecto): una identificación solo por prototipo las perdería.
- **{nonperov_acc} entradas que NO son perovskita según la conectividad sí coinciden** con el
  prototipo: serían falsos positivos de esa identificación.

## 4. Validación frente a Bartel et al. (2019), *in-sample*

> **Circularidad declarada:** el umbral de 135° y la regla del par libre se eligieron
> comparando con estas etiquetas. Esta tabla NO es una validación independiente; la
> independiente es la revisión manual de F1.4.

| Regla | Composiciones comparadas | Perovskitas detectadas | No perovskitas mal marcadas |
|---|---|---|---|
| Principal (135°, híbrida) | {m["n"]} | {m["tp"]}/{m["tp"] + m["fn"]} ({m["recall"]:.1%}) | {m["fp"]} ({m["fp_rate"]:.1%}): {", ".join(m["fp_list"])} |
| Sensibilidad (140°, estricta) | {st["n"]} | {st["tp"]}/{st["tp"] + st["fn"]} ({st["recall"]:.1%}) | {st["fp"]} ({st["fp_rate"]:.1%}): {", ".join(st["fp_list"])} |

- **Falsos positivos:** SrSiO₃ y MnTiO₃ (TiMnO₃) tienen en MP entradas experimentales con
  estructura de perovskita; Bartel los clasifica como no perovskita. Que sean polimorfos de
  alta presión es plausible, pero **no está verificado**: se comprueba en F1.4 y, si se
  confirma, reciben la bandera de alta presión (F1.5).
- **No detectadas ({m["fn"]}):** {", ".join(m["missed"])}. Incluyen yodatos (excluidos por
  definición: el I⁵⁺ forma pirámides IO₃, no octaedros); compuestos cuyo polimorfo
  perovskita no figura como experimental en MP (RbNbO₃, RbTaO₃, YGaO₃), y perovskitas muy
  distorsionadas a cuyo sitio A CrystalNN asigna NC 6 (LaErO₃, LaHoO₃, LaYO₃, ScCrO₃,
  ScBiO₃). Estas últimas quedan como casos límite para la revisión manual.

## 5. Conflictos Bartel–Gu (F1.2) según nuestra identificación

"""
        + pd.DataFrame(s["conflicts_bartel_gu"]).to_markdown(index=False)
        + f"""

## 6. Casos límite → validación manual (F1.4)

{len(border)} entradas ({int(border["experimental"].sum())} experimentales) en `borderline.csv`, ordenadas
con las experimentales primero.
"""
    )
    (OUT / "informe.md").write_text(md, "utf-8")
    print(
        json.dumps(
            {k: s[k] for k in ("entries_by_class", "bartel_validation_main")},
            indent=1,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":  # requisito en Windows (regla 2.2)
    main()
