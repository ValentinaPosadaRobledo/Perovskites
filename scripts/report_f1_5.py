"""Informe de F1.5: etiquetas por composición, deduplicación y banderas de sensibilidad.

Lee `data/interim/f1_5_labels.parquet` y `f1_5_entries.parquet` (generados por
`uv run python -m perovskite_pu.data.labeling`) y escribe en `results/f1_5_etiquetado/`:
    labels.csv     una fila por composición: etiqueta, sitios A/B, recuentos y banderas
    summary.json   cifras del informe
    informe.md     informe legible, generado (no editar a mano)

Uso:
    uv run python scripts/report_f1_5.py
"""

from __future__ import annotations

import json

import pandas as pd

from perovskite_pu.utils.hashing import sha256_file
from perovskite_pu.utils.paths import INTERIM_DIR, RESULTS_DIR
from perovskite_pu.utils.run_metadata import git_commit, git_is_dirty

OUT = RESULTS_DIR / "f1_5_etiquetado"
FLAGS = [
    "flag_high_pressure", "flag_metastable_dft", "flag_ambiguous_oxidation",
    "flag_bartel_conflict", "flag_only_hexagonal", "flag_icsd_without_exp",
]  # fmt: skip


def main() -> None:
    labels = pd.read_parquet(INTERIM_DIR / "f1_5_labels.parquet")
    entries = pd.read_parquet(INTERIM_DIR / "f1_5_entries.parquet")
    counts = labels["label"].value_counts()
    p, u, amb = (int(counts.get(k, 0)) for k in ("positive", "unlabeled", "ambiguous"))
    pos = labels[labels["label"] == "positive"]
    sens = pos[
        pos["flag_high_pressure"] | pos["flag_metastable_dft"] | pos["flag_ambiguous_oxidation"]
    ]
    dups = entries.groupby("dup_group").size()

    s = {
        "code_commit": git_commit(), "code_dirty": git_is_dirty(),
        "labels_parquet_sha256": sha256_file(INTERIM_DIR / "f1_5_labels.parquet"),
        "entries": len(entries), "unique_structures": int(entries["dup_group"].nunique()),
        "duplicate_groups": int((dups > 1).sum()), "redundant_entries": int((dups[dups > 1] - 1).sum()),
        "manual_reviewed_entries": int(entries["manual_reviewed"].sum()),
        "labels": counts.to_dict(),
        "pu_set": p + u, "labeled_prevalence": p / (p + u),
        "labeled_prevalence_ambiguous_as_positive": (p + amb) / (p + amb + u),
        "flags": {f: {"total": int(labels[f].sum()),
                      "by_label": labels.loc[labels[f], "label"].value_counts().to_dict(),
                      "compositions": sorted(labels.loc[labels[f], "reduced_formula"])} for f in FLAGS},
        "positives_in_sensitivity_exclusion": len(sens),
    }  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(s, indent=2, ensure_ascii=False) + "\n", "utf-8")
    labels.to_csv(OUT / "labels.csv", index=False, lineterminator="\n")

    flag_rows = "\n".join(
        f"| `{f}` | {v['total']} | {', '.join(f'{k}: {n}' for k, n in v['by_label'].items()) or '—'} | {', '.join(v['compositions'][:12])}{' …' if v['total'] > 12 else ''} |"
        for f, v in s["flags"].items()
    )
    md = f"""# Informe F1.5: etiquetas por composición

> Generado por `scripts/report_f1_5.py` (commit `{s["code_commit"][:7]}`{", árbol sucio" if s["code_dirty"] else ""}).
> No editar a mano. Tabla completa: `labels.csv`.

## 1. Deduplicación estructural (StructureMatcher, dentro de cada composición)

{s["entries"]} entradas → **{s["unique_structures"]} estructuras únicas** ({s["duplicate_groups"]} grupos con duplicados,
{s["redundant_entries"]} entradas redundantes). Con las tolerancias por defecto, StructureMatcher agrupa
variantes distorsionadas del mismo armazón (p. ej. una perovskita cúbica teórica con su
versión R3c experimental). **Las etiquetas usan todas las entradas de cada composición**,
así que la deduplicación no las afecta. Solo importará si F3.5 necesita una estructura
representativa por composición.

"Agrupar composiciones dopadas o casi idénticas con su padre" no aplica: MP solo contiene
ABO₃ ordenados y estequiométricos.

## 2. Etiquetas ({len(labels)} composiciones)

| Etiqueta | Composiciones | Uso |
|---|---|---|
| `positive` (P) | {p} | positivos del problema PU |
| `unlabeled` (U) | {u} | no etiquetados (NO son negativos) |
| `synthesized_non_perovskite` | {int(counts.get("synthesized_non_perovskite", 0))} | aparte: validación de formabilidad (F3.4) |
| `ambiguous` | {amb} | fuera de P y U; P en la sensibilidad (F3.6) |
| `polyanion` | {int(counts.get("polyanion", 0))} | excluidas |

- **Conjunto PU: {s["pu_set"]} composiciones.** Prevalencia de positivas etiquetadas:
  **{s["labeled_prevalence"]:.1%}** ({s["labeled_prevalence_ambiguous_as_positive"]:.1%} con las ambiguas como P).
  No es la prior π, que se estima en F3.2.
- Entradas con veredicto manual (F1.4): {s["manual_reviewed_entries"]}.

## 3. Banderas (análisis de sensibilidad; no cambian la etiqueta)

| Bandera | Total | Por etiqueta | Composiciones |
|---|---|---|---|
{flag_rows}

- `flag_high_pressure`: lista curada en `configs/data.yaml` (ampliable).
- `flag_metastable_dft`: la perovskita experimental queda ≥ 0,05 eV/átomo por encima del
  polimorfo más estable, que no es perovskita (E_hull GGA/GGA+U).
- La sensibilidad de F3.6 excluye **{s["positives_in_sensitivity_exclusion"]} positivas**: alta presión, metaestables u
  oxidación ambigua.
"""
    (OUT / "informe.md").write_text(md, "utf-8")
    print(
        json.dumps(
            {
                k: s[k]
                for k in (
                    "labels",
                    "pu_set",
                    "labeled_prevalence",
                    "positives_in_sensitivity_exclusion",
                )
            },
            indent=1,
        )
    )


if __name__ == "__main__":  # requisito en Windows (regla 2.2)
    main()
