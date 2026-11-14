"""Informe de F2.2: features `post_dft` (E_hull y ΔHf del polimorfo perovskita).

Lee las salidas de `perovskite_pu.features.post_dft` y `train_pool`. Las cifras de P y U son
SOLO de `train_pool`, así que el holdout no se toca. Las sintetizadas no perovskita y las
ambiguas se describen aparte.

Escribe en `results/f2_2_post_dft/`: summary.json, coverage.csv, informe.md.

Uso:
    uv run python -m perovskite_pu.features.post_dft   # primero, las features
    uv run python scripts/report_f2_2.py
"""

from __future__ import annotations

import json

import pandas as pd
import yaml
from scipy.stats import spearmanr

from perovskite_pu.utils.paths import (
    CONFIGS_DIR,
    INTERIM_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    RESULTS_DIR,
    ROOT,
    TRAIN_POOL_FILENAME,
)
from perovskite_pu.utils.run_metadata import git_commit, git_is_dirty

OUT = RESULTS_DIR / "f2_2_post_dft"
CLASS_NAMES = {
    "P": "positivas (train_pool)",
    "U": "no etiquetadas (train_pool)",
    "SNP": "sintetizadas no perovskita",
    "AMB": "ambiguas",
}


def iqr(v: pd.Series) -> str:
    """Mediana [Q1, Q3] de una serie, sin NaN."""
    v = v.dropna()
    if v.empty:
        return "—"
    return f"{v.median():.3f} [{v.quantile(0.25):.3f}, {v.quantile(0.75):.3f}]"


def main() -> None:
    cfg = yaml.safe_load((CONFIGS_DIR / "data.yaml").read_text("utf-8"))
    out_cfg, post_cfg = cfg["features"]["output"], cfg["features"]["post_dft"]
    table = pd.read_parquet(ROOT / out_cfg["post_table"])
    meta = pd.read_parquet(ROOT / out_cfg["post_metadata"])
    tp = pd.read_parquet(PROCESSED_DIR / TRAIN_POOL_FILENAME).set_index("composition")
    labels = pd.read_parquet(INTERIM_DIR / "f1_5_labels.parquet").set_index("reduced_formula")
    entries = pd.read_parquet(INTERIM_DIR / "f1_5_entries.parquet")
    thermo = pd.read_parquet(RAW_DIR / "mp_abo3_thermo.parquet")

    classes = {
        "P": tp.index[tp["label"] == 1],
        "U": tp.index[tp["label"] == 0],
        "SNP": labels.index[labels["label"] == "synthesized_non_perovskite"],
        "AMB": labels.index[labels["label"] == "ambiguous"],
    }

    # 1. Elegibilidad por clase.
    gga_types = thermo.loc[
        thermo["thermo_type"].str.lower() == post_cfg["thermo_type"].lower(), "material_id"
    ]
    has_gga = entries.assign(g=entries["material_id"].isin(set(gga_types)))
    has_gga = has_gga.groupby("reduced_formula")["g"].any()
    cov_rows = []
    for key, idx in classes.items():
        m = meta.loc[idx]
        cov_rows.append({
            "clase": key, "n": len(idx),
            "elegibles_GGA": int(m["eligible_post_dft"].sum()),
            "sin_termo_GGA": int((~has_gga.reindex(idx).fillna(False).astype(bool)).sum()),
            "con_GGA_sin_perovskita": int((has_gga.reindex(idx).fillna(False).astype(bool)
                                           & ~m["eligible_post_dft"]).sum()),
            "elegibles_r2SCAN": int(m["eligible_post_dft_r2scan"].sum()),
        })  # fmt: skip
    coverage = pd.DataFrame(cov_rows)

    # Elementos de las U sin termodinámica GGA (se esperaba Yb).
    u_no_gga = [c for c in classes["U"] if not bool(has_gga.get(c, False))]
    from pymatgen.core import Composition

    no_gga_elements = (
        pd.Series(
            [el.symbol for c in u_no_gga for el in Composition(c).elements if el.symbol != "O"]
        )
        .value_counts()
        .head(5)
        .to_dict()
    )
    # Polimorfos de las U con GGA pero sin entrada perovskita.
    u_no_pv = [c for c in classes["U"] if bool(has_gga.get(c, False))
               and not bool(meta.at[c, "eligible_post_dft"])]  # fmt: skip
    no_pv_classes = (
        entries[entries["reduced_formula"].isin(u_no_pv)]["cls_final"].value_counts().to_dict()
    )

    # 2. Circularidad: ¿la entrada elegida es la experimental? (solo elegibles)
    circ = {}
    for key in ("P", "U", "SNP", "AMB"):
        m = meta.loc[classes[key]]
        m = m[m["eligible_post_dft"]]
        circ[key] = float(m["experimental_gga"].astype(bool).mean()) if len(m) else None

    # 3. Polimorfo perovskita frente al estado fundamental: ΔE = E_hull(pv) − E_hull(fund.).
    gap_rows = []
    for key, idx in classes.items():
        m = meta.loc[idx]
        m = m[m["eligible_post_dft"]]
        de = table.loc[m.index, "thermo_ehull"] - m["ground_state_ehull_gga"]
        gap_rows.append({
            "clase": key, "elegibles": len(m),
            "perovskita_es_fundamental": f"{(de <= 1e-6).mean():.1%}" if len(m) else "—",
            "ΔE (eV/átomo)": iqr(de),
        })  # fmt: skip

    # 4. Distribución de E_hull y ΔHf (elegibles), por clase.
    dist_rows = []
    for key, idx in classes.items():
        ok = meta.loc[idx, "eligible_post_dft"]
        t = table.loc[ok.index[ok]]
        dist_rows.append({"clase": key, "E_hull (eV/átomo)": iqr(t["thermo_ehull"]),
                          "ΔHf (eV/átomo)": iqr(t["thermo_eform"])})  # fmt: skip

    # 5. r2SCAN frente a GGA en las composiciones de train_pool con ambas.
    both = table.loc[tp.index].dropna(subset=["thermo_ehull", "thermo_ehull_r2scan"])
    rho = spearmanr(both["thermo_ehull"], both["thermo_ehull_r2scan"]).statistic
    lab_both = tp.loc[both.index, "label"]
    on_hull = {"GGA": float((both["thermo_ehull"] < 1e-6).mean()),
               "r2SCAN": float((both["thermo_ehull_r2scan"] < 1e-6).mean())}  # fmt: skip
    rho_eform = spearmanr(both["thermo_eform"], both["thermo_eform_r2scan"]).statistic
    u_med = float(both.loc[lab_both.index[lab_both == 0], "thermo_ehull"].median())
    r2 = {"n": len(both), "P": int(lab_both.sum()), "U": int((lab_both == 0).sum()),
          "spearman_ehull": float(rho), "spearman_eform": float(rho_eform),
          "on_hull_frac": on_hull, "u_median_ehull_gga": u_med,
          "same_entry": float((meta.loc[both.index, "material_id_gga"]
                               == meta.loc[both.index, "material_id_r2scan"]).mean())}  # fmt: skip

    s = {
        "code_commit": git_commit(), "code_dirty": git_is_dirty(), "config": post_cfg,
        "coverage": cov_rows, "u_without_gga_top_elements": no_gga_elements,
        "u_with_gga_without_perovskite_entry_classes": no_pv_classes,
        "chosen_entry_experimental_frac": circ, "perovskite_vs_ground_state": gap_rows,
        "distribution": dist_rows, "r2scan_vs_gga": r2,
    }  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(s, indent=2, ensure_ascii=False) + "\n", "utf-8")
    coverage.to_csv(OUT / "coverage.csv", index=False, lineterminator="\n")

    cn = lambda df: df.assign(clase=df["clase"].map(CLASS_NAMES))  # noqa: E731
    cP, cU = cov_rows[0], cov_rows[1]
    md = f"""# Informe F2.2: features `post_dft`

> Generado por `scripts/report_f2_2.py` (commit `{s["code_commit"][:7]}`{", árbol sucio" if s["code_dirty"] else ""}).
> No editar a mano. Cifras de P y U solo sobre `train_pool` ({len(tp)} composiciones,
> {len(classes["P"])} P); el holdout no se toca.

## 1. Definición

`post_dft` = `pre_dft` + **E_hull** + **ΔHf** (por átomo) de la entrada de MP clasificada como
`perovskite` con **menor E_hull**, con una sola funcional (`{post_cfg["thermo_type"]}`).
Decisión de la autora (2026-11-14): el polimorfo perovskita y no el estado fundamental. La
sensibilidad 4 repite lo mismo con r2SCAN, solo en las composiciones que lo tienen.

**Composiciones no elegibles:** las que no tienen entrada perovskita con termodinámica GGA se
**excluyen** de `post_dft`; no se imputan. Todas las P tienen esa entrada, porque su estructura
experimental está en MP, y muchas U no: imputar daría la regla "sin polimorfo perovskita → U",
derivada de la etiqueta. `perovskite_pu.features.sets` rechaza pedirlas.

## 2. Elegibilidad

{cn(coverage).to_markdown(index=False)}

- **`post_dft` sobre `train_pool`: {cP["elegibles_GGA"]} P y {cU["elegibles_GGA"]} U.** Sus comparaciones pareadas con
  `pre_dft` y la línea base de solo E_hull usan este mismo subconjunto.
- U sin ninguna termodinámica GGA: {cU["sin_termo_GGA"]}; elementos más frecuentes: {no_gga_elements}. Son
  compuestos hipotéticos con Yb que en MP solo tienen r2SCAN (decisión: se excluyen).
- U con GGA pero sin entrada `perovskite`: {cU["con_GGA_sin_perovskita"]}; clases de sus entradas, en n.º de entradas: {no_pv_classes}.

## 3. Circularidad: la entrada elegida es la experimental

Fracción de composiciones elegibles cuya perovskita de menor energía es una entrada
experimental: P {circ["P"]:.1%} · U {circ["U"]:.1%} · sintetizadas no perovskita {circ["SNP"] if circ["SNP"] is None else f"{circ['SNP']:.1%}"}.

En las P, la feature es casi siempre la E_hull de la estructura experimental (relajada, con sus
inclinaciones); en las U es la de un prototipo hipotético. Además, el hull de MP lo definen en
parte las fases de ICSD. Por eso `post_dft` es una **cota superior informada por el hull**
(regla 2.1.8) y se acompaña de la línea base de solo E_hull (F2.4). Se audita en F2.3.

## 4. Polimorfo perovskita frente al estado fundamental (GGA)

ΔE = E_hull(perovskita) − E_hull(estado fundamental), en eV/átomo (mediana [Q1, Q3]).

{cn(pd.DataFrame(gap_rows)).to_markdown(index=False)}

## 5. Distribución de E_hull y ΔHf (elegibles; mediana [Q1, Q3])

{cn(pd.DataFrame(dist_rows)).to_markdown(index=False)}

Es solo descriptiva: la evaluación de la E_hull como clasificador (línea base) se hace en F2.4
con los folds de cada esquema.

## 6. Sensibilidad 4: r2SCAN

Composiciones de `train_pool` con perovskita en GGA y en r2SCAN: {r2["n"]} ({r2["P"]} P, {r2["U"]} U).
Solo ellas son elegibles para `post_dft_r2scan`, porque se compara con GGA en el mismo
subconjunto. El análisis es **descriptivo** (decisión de la autora, 2026-11-14) y el subconjunto
está sesgado: además de tener pocas U, estas son casi estables (mediana de su E_hull GGA:
{r2["u_median_ehull_gga"]:.3f} eV/átomo), porque MP calculó r2SCAN sobre todo para fases estables.

- Spearman de E_hull GGA frente a r2SCAN: {r2["spearman_ehull"]:.3f}. Es bajo por los empates en 0: sobre
  el hull hay un {r2["on_hull_frac"]["GGA"]:.0%} con GGA y un {r2["on_hull_frac"]["r2SCAN"]:.0%} con r2SCAN. En ΔHf, ρ = {r2["spearman_eform"]:.3f}.
- Misma entrada elegida con ambas funcionales: {r2["same_entry"]:.1%}.
"""
    (OUT / "informe.md").write_text(md, "utf-8")
    print(json.dumps({k: s[k] for k in ("coverage", "chosen_entry_experimental_frac", "r2scan_vs_gga")},
                     indent=1, ensure_ascii=False))  # fmt: skip


if __name__ == "__main__":  # requisito en Windows (regla 2.2)
    main()
