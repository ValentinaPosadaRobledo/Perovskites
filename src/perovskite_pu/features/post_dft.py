"""F2.2: features `post_dft` = `pre_dft` + E_hull + ΔHf del polimorfo perovskita (regla 2.1.8).

Definición (configs/data.yaml, `features.post_dft`):
- Por composición, la E_hull y la ΔHf (por átomo) de la entrada de MP clasificada como
  `perovskite` en F1.5 con MENOR E_hull, con una sola funcional (GGA_GGA+U). Decisión de la
  autora (2026-11-14): el polimorfo perovskita y no el estado fundamental, porque la pregunta
  es si la perovskita es sintetizable.
- Sensibilidad 4: lo mismo con r2SCAN, solo en las composiciones elegibles con AMBAS
  funcionales, para comparar con GGA en el mismo subconjunto (análisis descriptivo).

Composiciones sin entrada perovskita con termodinámica de esa funcional: se EXCLUYEN de
`post_dft` y NO se imputan. Todas las positivas tienen esa entrada, porque su estructura
experimental está en MP; muchas no etiquetadas no. Imputar daría la regla "sin polimorfo
perovskita → U", que viene de la etiqueta. Por eso cada composición lleva una bandera de
elegibilidad, y `post_dft` se entrena y evalúa solo con las elegibles.

Circularidad, que se declara: en las positivas, la perovskita de menor energía es casi siempre
la entrada EXPERIMENTAL (estructura relajada, con sus inclinaciones), y en las no etiquetadas es
un prototipo hipotético. Además, el hull de MP lo definen en parte las fases de ICSD. Por eso
`post_dft` es una cota superior informada por el hull, no un modelo de sintetizabilidad.

Las columnas de metadatos (entrada elegida, si es experimental, número de entradas) se guardan
en un archivo APARTE: no son features y dependen de la etiqueta.

Uso:
    uv run python -m perovskite_pu.features.post_dft
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import pandas as pd

from perovskite_pu.utils.paths import CONFIGS_DIR, INTERIM_DIR, RAW_DIR, ROOT

# Nombre de cada feature termodinámica en la tabla (prefijo `thermo_`).
THERMO_COLUMNS = {
    "ehull": "thermo_ehull",
    "eform": "thermo_eform",
    "ehull_r2scan": "thermo_ehull_r2scan",
    "eform_r2scan": "thermo_eform_r2scan",
}


def perovskite_polymorph_thermo(
    entries: pd.DataFrame, thermo: pd.DataFrame, thermo_type: str, classes: Sequence[str]
) -> pd.DataFrame:
    """Entrada perovskita de menor E_hull de cada composición con una funcional.

    `entries` es `f1_5_entries` (con `cls_final` y `experimental`) y `thermo` la tabla del
    endpoint `thermo`. El tipo se compara sin distinguir mayúsculas (la API acepta `R2SCAN` y
    devuelve `r2SCAN`). Los empates en E_hull se rompen por `material_id`, para que sea
    determinista. Devuelve una fila por composición que TIENE esa entrada.
    """
    th = thermo[thermo["thermo_type"].str.lower() == thermo_type.lower()]
    e = entries.loc[entries["cls_final"].isin(classes), ["material_id", "reduced_formula",
                                                          "experimental"]]  # fmt: skip
    e = e.merge(
        th[["material_id", "energy_above_hull", "formation_energy_per_atom"]], on="material_id"
    ).dropna(subset=["energy_above_hull", "formation_energy_per_atom"])
    e = e.sort_values(["reduced_formula", "energy_above_hull", "material_id"])
    best = e.drop_duplicates("reduced_formula", keep="first").set_index("reduced_formula")
    best["n_perovskite_entries"] = e.groupby("reduced_formula").size()
    return best


def ground_state_ehull(entries: pd.DataFrame, thermo: pd.DataFrame, thermo_type: str) -> pd.Series:
    """Mínima E_hull entre TODAS las entradas de cada composición (solo para el informe)."""
    th = thermo[thermo["thermo_type"].str.lower() == thermo_type.lower()]
    e = entries[["material_id", "reduced_formula"]].merge(
        th[["material_id", "energy_above_hull"]], on="material_id"
    )
    return e.groupby("reduced_formula")["energy_above_hull"].min()


def build_post_dft(
    formulas: Iterable[str],
    entries: pd.DataFrame,
    thermo: pd.DataFrame,
    post_cfg: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tabla de features termodinámicas (NaN si no hay entrada) y metadatos con elegibilidad."""
    if post_cfg.get("missing_policy", "exclude") != "exclude":
        # Única política admitida: imputar crearía un atajo derivado de la etiqueta.
        raise ValueError("features.post_dft.missing_policy solo admite 'exclude'.")
    idx = pd.Index(sorted(set(formulas)), name="reduced_formula")
    classes = post_cfg["perovskite_classes"]
    table, meta = pd.DataFrame(index=idx), pd.DataFrame(index=idx)
    for suffix, ttype in (("", post_cfg["thermo_type"]),
                          ("_r2scan", post_cfg["sensitivity_thermo_type"])):  # fmt: skip
        best = perovskite_polymorph_thermo(entries, thermo, ttype, classes).reindex(idx)
        table[THERMO_COLUMNS["ehull" + suffix]] = best["energy_above_hull"].astype(float)
        table[THERMO_COLUMNS["eform" + suffix]] = best["formation_energy_per_atom"].astype(float)
        tag = suffix or "_gga"
        meta["material_id" + tag] = best["material_id"]
        meta["experimental" + tag] = best["experimental"].astype("boolean")
        meta["n_perovskite_entries" + tag] = best["n_perovskite_entries"].fillna(0).astype(int)
        meta["eligible_post_dft" + suffix] = best["energy_above_hull"].notna()
    # Sensibilidad 4: r2SCAN se compara con GGA en el MISMO subconjunto, así que exige ambas.
    # Las composiciones con Yb solo tienen r2SCAN y quedan fuera de las dos familias.
    meta["eligible_post_dft_r2scan"] &= meta["eligible_post_dft"]
    meta["ground_state_ehull_gga"] = ground_state_ehull(
        entries, thermo, post_cfg["thermo_type"]
    ).reindex(idx)
    return table, meta


def family_columns_post(
    families_cfg: Mapping[str, Any], pre_families: Mapping[str, list[str]]
) -> dict[str, list[str]]:
    """Columnas de cada familia `post_dft*`: las de la familia que extiende + las termodinámicas."""
    return {
        fam: list(pre_families[spec["extends"]]) + [THERMO_COLUMNS[t] for t in spec["thermo"]]
        for fam, spec in families_cfg.items()
    }


def main(argv: list[str] | None = None) -> None:
    """Calcula las features termodinámicas de todas las composiciones no polianiónicas."""
    import yaml

    cfg = yaml.safe_load((CONFIGS_DIR / "data.yaml").read_text("utf-8"))
    feat_cfg, out = cfg["features"], cfg["features"]["output"]
    parser = argparse.ArgumentParser(description="F2.2: features post_dft por composición.")
    parser.add_argument("--labels", default=str(INTERIM_DIR / "f1_5_labels.parquet"))
    args = parser.parse_args(argv)

    labels = pd.read_parquet(args.labels)
    formulas = labels.loc[labels["label"] != "polyanion", "reduced_formula"]
    entries = pd.read_parquet(INTERIM_DIR / "f1_5_entries.parquet")
    thermo = pd.read_parquet(RAW_DIR / "mp_abo3_thermo.parquet")
    table, meta = build_post_dft(formulas, entries, thermo, feat_cfg["post_dft"])

    # Las familias post_dft extienden las de pre_dft: hacen falta sus columnas.
    pre_cols = json.loads((ROOT / out["columns"]).read_text("utf-8"))["families"]
    columns = {
        "families": family_columns_post(feat_cfg["families_post_dft"], pre_cols),
        # Columna de metadatos que indica qué composiciones admite cada familia.
        "eligibility": {"post_dft": "eligible_post_dft",
                        "post_dft_r2scan": "eligible_post_dft_r2scan"},
    }  # fmt: skip
    table.to_parquet(ROOT / out["post_table"])
    meta.to_parquet(ROOT / out["post_metadata"])
    (ROOT / out["post_columns"]).write_text(
        json.dumps(columns, indent=2, ensure_ascii=False) + "\n", "utf-8"
    )
    print(f"{len(table)} composiciones -> {out['post_table']}")
    for fam, col in columns["eligibility"].items():
        print(
            f"  {fam}: {int(meta[col].sum())} elegibles, {len(columns['families'][fam])} columnas"
        )


if __name__ == "__main__":  # requisito en Windows (regla 2.2)
    main()
