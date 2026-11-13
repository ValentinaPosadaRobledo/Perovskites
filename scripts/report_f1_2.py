"""Informe de F1.2: datos de Gu et al. y Bartel et al., y su cruce con MP 2026.

Genera en `results/f1_2_datos_externos/`:
    summary.json   todas las cifras del informe (legible por código)
    conflicts.csv  óxidos ABO3 con etiquetas en conflicto entre Bartel y Gu
    informe.md     el informe legible, generado (no editar a mano)

Uso:
    uv run python scripts/report_f1_2.py
"""

from __future__ import annotations

import json

import pandas as pd
import yaml

from perovskite_pu.data.external import (
    BARTEL_DIR,
    composition_labels,
    gu_root,
    gu_test_composition_leakage,
    label_conflicts,
    load_bartel2019,
    load_gu2022,
    to_legacy_mpid,
)
from perovskite_pu.utils.paths import CONFIGS_DIR, DATA_DIR, RAW_DIR, RESULTS_DIR
from perovskite_pu.utils.run_metadata import git_commit, git_is_dirty

OUT = RESULTS_DIR / "f1_2_datos_externos"


def crosstab_md(a: pd.Series, b: pd.Series) -> str:
    """Tabla cruzada en Markdown, con totales."""
    return pd.crosstab(a, b, margins=True, margins_name="Total").to_markdown()


def main() -> None:
    cfg = yaml.safe_load((CONFIGS_DIR / "data.yaml").read_text("utf-8"))["external_sources"]
    manifest = json.loads((DATA_DIR / "MANIFEST.json").read_text("utf-8"))

    # --- Carga ---------------------------------------------------------------
    gu = load_gu2022(gu_root(), cfg["gu2022"]["split_index"])
    bartel = load_bartel2019(BARTEL_DIR / cfg["bartel2019"]["table"])
    mp = pd.read_parquet(RAW_DIR / "mp_abo3_index.parquet")
    mp["legacy_id"] = mp["material_id"].map(to_legacy_mpid)
    gu_ox, b_ox = gu[gu["abo3_oxide"]], bartel[bartel["abo3_oxide"]]

    # --- Gu: conteos y fuga entre polimorfos -------------------------------------
    s: dict = {
        "code_commit": git_commit(), "code_dirty": git_is_dirty(),
        "mp_database_version": manifest.get("mp_database_version"),
        "gu": {
            "crystals": len(gu), "positives": int(gu["is_positive"].sum()),
            "compositions": int(gu["reduced_formula"].nunique()),
            "abo3_crystals": len(gu_ox), "abo3_positive_crystals": int(gu_ox["is_positive"].sum()),
            "abo3_compositions": int(gu_ox["reduced_formula"].nunique()),
            "abo3_positive_compositions": int(gu_ox.loc[gu_ox["is_positive"], "reduced_formula"].nunique()),
            "test_leakage_all": gu_test_composition_leakage(gu),
            "test_leakage_abo3": gu_test_composition_leakage(gu, only_abo3=True),
        },
    }  # fmt: skip

    # --- Gu vs MP 2026 -------------------------------------------------------------
    legacy = set(mp["legacy_id"])
    with_mp = gu_ox[gu_ox["mp_legacy_ids"].map(len) > 0]
    found = with_mp["mp_legacy_ids"].map(lambda ids: any(i in legacy for i in ids))
    mp_exp = mp.groupby("reduced_formula")["theoretical"].apply(lambda t: (~t.astype(bool)).any())
    gu_pos = composition_labels(gu_ox, "is_positive")
    both = gu_pos.index.intersection(mp_exp.index)
    s["gu_vs_mp"] = {
        "abo3_crystals_with_mp_id": len(with_mp),
        "abo3_crystals_with_mp_id_found_in_mp2026": int(found.sum()),
        "abo3_compositions_shared": len(both),
        "mp_compositions_not_in_gu": len(set(mp_exp.index) - set(gu_pos.index)),
        "shared_gu_positive_mp_experimental": int((gu_pos[both] & mp_exp[both]).sum()),
        "shared_gu_virtual_mp_experimental": int((~gu_pos[both] & mp_exp[both]).sum()),
        "shared_gu_positive_mp_theoretical_only": int((gu_pos[both] & ~mp_exp[both]).sum()),
    }

    # --- Bartel vs MP y vs Gu ------------------------------------------------------------
    b = b_ox.set_index("reduced_formula")
    conflicts = label_conflicts(bartel, gu)
    s["bartel"] = {
        "compounds": len(bartel), "abo3_oxides": len(b_ox),
        "abo3_perovskite": int(b_ox["is_perovskite"].sum()),
        "abo3_non_perovskite": int((~b_ox["is_perovskite"]).sum()),
        "abo3_in_mp2026": int(b.index.isin(mp_exp.index).sum()),
        "perovskite_but_mp_theoretical_only": int(
            (b["is_perovskite"] & b.index.map(lambda c: c in mp_exp.index and not mp_exp[c])).sum()
        ),
        "conflicts_bartel_no_gu_si": int((conflicts["type"] == "bartel_no_gu_si").sum()),
        "conflicts_bartel_si_gu_virtual": int((conflicts["type"] == "bartel_si_gu_virtual").sum()),
    }  # fmt: skip

    # --- Escritura ---------------------------------------------------------------------------
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(s, indent=2, ensure_ascii=False) + "\n", "utf-8")
    conflicts.to_csv(OUT / "conflicts.csv", index=False, lineterminator="\n")

    lab_b = b_ox["is_perovskite"].map({True: "Bartel: perovskita", False: "Bartel: NO perovskita"})
    lab_mp = b_ox["reduced_formula"].map(
        lambda c: "no está en MP" if c not in mp_exp.index else ("MP: experimental" if mp_exp[c] else "MP: solo teórica")
    )  # fmt: skip
    lab_gu = b_ox["reduced_formula"].map(
        lambda c: "no está en Gu" if c not in gu_pos.index else ("Gu: positivo" if gu_pos[c] else "Gu: virtual")
    )  # fmt: skip
    g, gm, bt = s["gu"], s["gu_vs_mp"], s["bartel"]
    la, lo = g["test_leakage_all"], g["test_leakage_abo3"]
    by_type = conflicts.groupby("type")["reduced_formula"].apply(lambda x: ", ".join(x))
    md = f"""# Informe F1.2: datos externos y su cruce con MP

> Generado por `scripts/report_f1_2.py` (commit `{s["code_commit"][:7]}`{", árbol sucio" if s["code_dirty"] else ""}).
> No editar a mano. MP {s["mp_database_version"]}. Cifras en `summary.json`.

## 1. Gu et al. (2022), Zenodo {cfg["gu2022"]["zenodo_record"]}

| | Cristales | Positivos | Composiciones |
|---|---|---|---|
| Todas las clases | {g["crystals"]} | {g["positives"]} | {g["compositions"]} |
| Óxidos ABO₃ | {g["abo3_crystals"]} | {g["abo3_positive_crystals"]} | {g["abo3_compositions"]} ({g["abo3_positive_compositions"]} con algún positivo) |

**Fuga entre polimorfos en su test** (dato de partida de S2, preregistro §10b). El test de
positivos es idéntico en sus 100 particiones.

| Subconjunto | Positivos en test | Composición también en train/val | Composición entre positivos de train/val |
|---|---|---|---|
| Todas las clases | {la["test_positives"]} | {la["composition_in_train_or_val"]} ({la["composition_in_train_or_val_frac"]:.0%}) | {la["composition_among_train_val_positives"]} ({la["composition_among_train_val_positives_frac"]:.0%}) |
| Óxidos ABO₃ | {lo["test_positives"]} | {lo["composition_in_train_or_val"]} ({lo["composition_in_train_or_val_frac"]:.0%}) | {lo["composition_among_train_val_positives"]} ({lo["composition_among_train_val_positives_frac"]:.0%}) |

## 2. Gu et al. frente a MP 2026

- **IDs:** MP pasó a IDs alfabéticos; se convirtieron al formato numérico de Gu
  (`to_legacy_mpid`). Cristales ABO₃ de Gu con ID de MP: {gm["abo3_crystals_with_mp_id"]};
  encontrados en MP 2026: {gm["abo3_crystals_with_mp_id_found_in_mp2026"]}.
- **Composiciones:** {gm["abo3_compositions_shared"]} compartidas. Composiciones de nuestra
  descarga ausentes en Gu: {gm["mp_compositions_not_in_gu"]}. Gu reúne además OQMD y AFLOW, así
  que tiene muchas más composiciones.
- **Etiquetas por composición** (compartidas): Gu positivo y MP experimental:
  {gm["shared_gu_positive_mp_experimental"]}. Gu virtual pero MP experimental:
  **{gm["shared_gu_virtual_mp_experimental"]}**. Gu positivo pero MP solo teórica:
  {gm["shared_gu_positive_mp_theoretical_only"]}.
- Las "Gu virtual, MP experimental" pueden ser polimorfos sintetizados **no perovskita** (la
  categoría aparte del proyecto) o síntesis posteriores a 2020. Se distinguen en F1.3.

## 3. Bartel et al. (2019), TableS1 (commit `{cfg["bartel2019"]["commit"][:7]}`)

{bt["compounds"]} compuestos; **{bt["abo3_oxides"]} óxidos ABO₃** ({bt["abo3_perovskite"]} perovskita,
{bt["abo3_non_perovskite"]} no perovskita); {bt["abo3_in_mp2026"]} están en MP 2026.

**Frente a MP 2026:**

{crosstab_md(lab_b.rename("Bartel"), lab_mp.rename("MP 2026"))}

**{bt["perovskite_but_mp_theoretical_only"]} perovskitas experimentales según Bartel figuran en MP solo
como teóricas:** con nuestra definición quedarían como no etiquetadas. Es una medida directa
de positivos ocultos en U.

**Frente a Gu et al.:**

{crosstab_md(lab_b.rename("Bartel"), lab_gu.rename("Gu"))}

## 4. Conflictos de etiqueta entre Bartel y Gu (`conflicts.csv`)

- **Bartel: NO perovskita; Gu: positivo ({bt["conflicts_bartel_no_gu_si"]}):**
  {by_type.get("bartel_no_gu_si", "—")}
- **Bartel: perovskita; Gu: solo virtual ({bt["conflicts_bartel_si_gu_virtual"]}):**
  {by_type.get("bartel_si_gu_virtual", "—")}

El primer grupo incluye polianiones (carbonatos, selenitos, teluritos, bromatos, cloratos,
silicatos) e ilmenitas (FeTiO₃, MnTiO₃), que no son perovskitas. Es coherente con que la
identificación por similitud a un prototipo acepte falsos positivos, lo que motiva la regla
2.1.1. **Es un conflicto entre fuentes, no un error demostrado:** se verifica con las
estructuras en F1.3/F1.4, donde estos compuestos entran como casos límite.

## 5. Implicaciones

1. **Formabilidad (F3.4):** antes de usar los negativos de Bartel hay que excluir los que
   nuestra identificación marque como perovskita (regla 2.3). Los {bt["conflicts_bartel_no_gu_si"]}
   conflictos del primer grupo se revisan a mano.
2. **PU:** las {bt["perovskite_but_mp_theoretical_only"]} perovskitas de Bartel que en MP son solo
   teóricas permiten estimar cuántos positivos hay escondidos en U. Sirve de contraste para la
   prior π (F3.2).
3. **S2 (F3.5):** la fuga entre polimorfos en el test de Gu es sustancial
   ({la["composition_in_train_or_val_frac"]:.0%} en todas las clases), lo que confirma que S2 es necesario.
"""
    (OUT / "informe.md").write_text(md, "utf-8")
    print(json.dumps(s, indent=2, ensure_ascii=False))


if __name__ == "__main__":  # requisito en Windows (regla 2.2)
    main()
