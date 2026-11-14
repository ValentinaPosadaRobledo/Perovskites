"""Informe de F2.1: features `pre_dft` (cobertura, radios sustitutos y solapamiento con Bartel).

Lee las salidas de `perovskite_pu.features.pre_dft` y `train_pool`. Las cifras de P y U son
SOLO de `train_pool`, así que el holdout no se toca. Las sintetizadas no perovskita y las
ambiguas, que no están en el pool PU, se describen aparte (validación de formabilidad).

Escribe en `results/f2_1_features/`: summary.json, coverage.csv, informe.md.

Uso:
    uv run python -m perovskite_pu.features.pre_dft   # primero, las features
    uv run python scripts/report_f2_1.py
"""

from __future__ import annotations

import json

import pandas as pd
import yaml

from perovskite_pu.features.pre_dft import GEOMETRIC_COLUMNS
from perovskite_pu.utils.paths import (
    CONFIGS_DIR,
    INTERIM_DIR,
    PROCESSED_DIR,
    RESULTS_DIR,
    ROOT,
    TRAIN_POOL_FILENAME,
)
from perovskite_pu.utils.run_metadata import git_commit, git_is_dirty

OUT = RESULTS_DIR / "f2_1_features"
CLASS_NAMES = {
    "P": "positivas (train_pool)",
    "U": "no etiquetadas (train_pool)",
    "SNP": "sintetizadas no perovskita",
    "AMB": "ambiguas",
}


def frac(mask: pd.Series) -> float:
    """Fracción de True (0 si la serie está vacía)."""
    return float(mask.mean()) if len(mask) else 0.0


def nan_reasons(notes: pd.Series) -> dict[str, int]:
    """Motivos por los que t/μ/τ no están definidos (se ignoran las notas de radio sustituto)."""
    exploded = notes.str.split(";").explode()
    exploded = exploded[(exploded != "") & ~exploded.str.contains("sustituto", na=False)]
    return {str(k): int(v) for k, v in exploded.value_counts().items()}


def iqr(v: pd.Series) -> str:
    """Mediana [Q1, Q3] de una serie, sin NaN."""
    v = v.dropna()
    if v.empty:
        return "—"
    return f"{v.median():.3f} [{v.quantile(0.25):.3f}, {v.quantile(0.75):.3f}]"


def main() -> None:
    cfg = yaml.safe_load((CONFIGS_DIR / "data.yaml").read_text("utf-8"))
    out_cfg = cfg["features"]["output"]
    table = pd.read_parquet(ROOT / out_cfg["table"])
    meta = pd.read_parquet(ROOT / out_cfg["metadata"])
    columns = json.loads((ROOT / out_cfg["columns"]).read_text("utf-8"))
    tp = pd.read_parquet(PROCESSED_DIR / TRAIN_POOL_FILENAME).set_index("composition")
    labels = pd.read_parquet(INTERIM_DIR / "f1_5_labels.parquet").set_index("reduced_formula")

    # Clases: P y U solo de train_pool; SNP y ambiguas, todas (no forman parte del pool PU).
    classes = {
        "P": tp.index[tp["label"] == 1],
        "U": tp.index[tp["label"] == 0],
        "SNP": labels.index[labels["label"] == "synthesized_non_perovskite"],
        "AMB": labels.index[labels["label"] == "ambiguous"],
    }
    geom = list(GEOMETRIC_COLUMNS.values())

    # 1. Columnas por grupo y por familia; columnas constantes en train_pool (las quita el
    #    umbral de varianza del Pipeline, ajustado en cada fold).
    X_tp = table.loc[tp.index]
    const = [c for c in X_tp.columns if X_tp[c].nunique(dropna=True) <= 1]
    fam_rows = []
    for fam, cols in columns["families"].items():
        n_const = sum(c in const for c in cols)
        fam_rows.append({"familia": fam, "columnas": len(cols), "constantes en train_pool": n_const,
                         "útiles": len(cols) - n_const})  # fmt: skip
    group_counts = {g: len(c) for g, c in columns["groups"].items()}

    # 2. Cobertura por clase.
    cov_rows = []
    for key, idx in classes.items():
        m, x = meta.loc[idx], table.loc[idx]
        cov_rows.append({
            "clase": key, "n": len(idx),
            "t_mu_definidos": frac(x["geom_t"].notna()), "tau_definido": frac(x["geom_tau"].notna()),
            "r_A_sustituto": frac(m["r_A_substitute"]), "r_B_sustituto": frac(m["r_B_substitute"]),
            "en_TableS1": frac(m["in_bartel2019"]),
        })  # fmt: skip
    coverage = pd.DataFrame(cov_rows)

    # 3. Motivos de NaN por clase.
    reasons = {k: nan_reasons(meta.loc[idx, "bartel_notes"]) for k, idx in classes.items()}

    # 4. Radios sustitutos más frecuentes en train_pool (elemento, oxidación, NC usado).
    m_tp = meta.loc[tp.index]
    sub = m_tp[m_tp["r_A_substitute"]]
    top_sub = (
        sub.groupby(["bartel_A", "bartel_nA", "bartel_A_cn"]).size().sort_values(ascending=False)
    )
    top_sub_rows = [
        {"A": a, "oxidación": int(n), "NC usado": int(cn), "composiciones": int(v)}
        for (a, n, cn), v in top_sub.head(15).items()
    ]
    sub_by_label = {
        "P": frac(m_tp.loc[classes["P"], "r_A_substitute"]),
        "U": frac(m_tp.loc[classes["U"], "r_A_substitute"]),
    }

    # 5. Sitios de Bartel (solo fórmula) frente a los de la estructura, en las P.
    p = m_tp.loc[classes["P"]].join(tp[["A", "B"]])
    disagree = p[p["bartel_A"] != p["A"]]
    site_agreement = {
        "agreement": frac(p["bartel_A"] == p["A"]),
        "disagreements": [{"composition": c, "A_estructura": r["A"], "A_bartel": r["bartel_A"]}
                          for c, r in disagree.iterrows()],
    }  # fmt: skip

    # 6. Solapamiento con la TableS1 (τ in-sample).
    in_s1 = m_tp["in_bartel2019"]
    u_in = m_tp.loc[classes["U"]]
    u_in = u_in[u_in["in_bartel2019"]]
    overlap = {
        "P_in_TableS1": int(in_s1.loc[classes["P"]].sum()),
        "P_outside_TableS1": int((~in_s1.loc[classes["P"]]).sum()),
        "U_in_TableS1": len(u_in),
        "U_outside_TableS1": int((~in_s1.loc[classes["U"]]).sum()),
        "U_in_TableS1_by_exp_label": {
            str(int(k)): int(v) for k, v in u_in["bartel2019_exp_label"].value_counts().items()
        },
        "P_in_TableS1_non_perovskite_for_bartel": int(
            (m_tp.loc[classes["P"], "bartel2019_exp_label"] == -1).sum()
        ),
    }

    # 7. Dos definiciones de balance de carga: `compound possible` de matminer (estados
    #    COMUNES de pymatgen) y `flag_no_charge_balance` de F1.5 (`oxi_state_guesses`, que usa
    #    los estados OBSERVADOS EN ICSD y, si no hay, los comunes).
    cb = pd.DataFrame({
        "label": tp["label"], "common": table.loc[tp.index, "compound possible"] == 1,
        "icsd": ~tp["flag_no_charge_balance"].astype(bool),
    })  # fmt: skip
    charge_balance = {
        lab: {"common_states_fail": int((~g["common"]).sum()),
              "icsd_states_fail": int((~g["icsd"]).sum()), "n": len(g)}
        for lab, g in (("P", cb[cb["label"] == 1]), ("U", cb[cb["label"] == 0]))
    }  # fmt: skip

    # 8. Distribución descriptiva de t, μ y τ por clase (mediana [Q1, Q3]); no es una evaluación.
    dist_rows = [
        {"clase": k, **{c: iqr(table.loc[idx, c]) for c in geom}} for k, idx in classes.items()
    ]
    dist = pd.DataFrame(dist_rows)

    s = {
        "code_commit": git_commit(), "code_dirty": git_is_dirty(),
        "n_compositions": len(table), "groups": group_counts, "families": fam_rows,
        "constant_columns_train_pool": const, "coverage": cov_rows, "nan_reasons": reasons,
        "r_A_substitute_by_label": sub_by_label, "r_A_substitute_top": top_sub_rows,
        "site_agreement_positives": site_agreement, "table_s1_overlap": overlap,
        "charge_balance": charge_balance,
    }  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(s, indent=2, ensure_ascii=False) + "\n", "utf-8")
    coverage.to_csv(OUT / "coverage.csv", index=False, lineterminator="\n")

    pct = coverage.copy()
    for c in ("t_mu_definidos", "tau_definido", "r_A_sustituto", "r_B_sustituto", "en_TableS1"):
        pct[c] = pct[c].map(lambda v: f"{v:.1%}")
    pct["clase"] = pct["clase"].map(CLASS_NAMES)
    reason_lines = "\n".join(
        f"- **{CLASS_NAMES[k]}:** " + ("; ".join(f"{r} ({n})" for r, n in v.items()) or "ninguno")
        for k, v in reasons.items()
    )
    dis = ", ".join(f"{d['composition']} (estructura: A = {d['A_estructura']}; Bartel: A = {d['A_bartel']})"
                    for d in site_agreement["disagreements"]) or "ninguna"  # fmt: skip
    cbp, cbu = charge_balance["P"], charge_balance["U"]
    md = f"""# Informe F2.1: features `pre_dft`

> Generado por `scripts/report_f2_1.py` (commit `{s["code_commit"][:7]}`{", árbol sucio" if s["code_dirty"] else ""}).
> No editar a mano. Cifras de P y U solo sobre `train_pool` ({len(tp)} composiciones,
> {len(classes["P"])} P); el holdout no se toca.

## 1. Familias de features

Se calculan una vez por composición ({len(table)} composiciones no polianiónicas), porque son
transformaciones fijas de la fórmula. La imputación, el umbral de varianza y el escalado van
dentro del Pipeline de cada fold.

Columnas por grupo: {", ".join(f"`{g}` {n}" for g, n in group_counts.items())}.

{pd.DataFrame(fam_rows).to_markdown(index=False)}

- **`pre_dft` (principal):** MAGPIE completo (Ward et al., 2016) + t + μ, **sin τ** (decisión de
  la autora, 2026-11-14).
- **`pre_dft_tau` (sensibilidad 14):** `pre_dft` + τ. Se evalúa en todo el conjunto (declarada
  *in-sample*) y solo sobre las composiciones fuera de la TableS1 (regla 2.1.6).
- Las {len(const)} columnas constantes en `train_pool` son las de `Stoichiometry` (todas las
  composiciones son ABO₃) y varias estadísticas `mode`/`maximum` de MAGPIE que siempre toma
  el O. No aportan información; las elimina el umbral de varianza del Pipeline.

## 2. t, μ y τ: algoritmo y cobertura

Reimplementación del algoritmo de Bartel et al. (2019) (`features/tolerance.py`), validada al
100 % contra los 576 compuestos de su TableS1: mismos A, B, estados de oxidación y radios, y
|Δt| y |Δτ| ≤ 0,005, que es el redondeo de la tabla (`tests/test_tolerance.py`). Los sitios
y la oxidación salen **solo de la fórmula**. Las columnas A/B del pool no se usan, porque salen
de la estructura y en las positivas dependen de la etiqueta.

{pct.to_markdown(index=False)}

**Motivos por los que t, μ o τ no están definidos** (quedan NaN y se imputan en el Pipeline):

{reason_lines}

La ausencia de t y μ está **muy asociada a la etiqueta**: falta en el
{1 - cov_rows[1]["t_mu_definidos"]:.1%} de las U y en el {1 - cov_rows[0]["t_mu_definidos"]:.1%} de las P. Equivale a "sin combinación de
oxidación de Bartel que equilibre la carga". No es fuga, porque es una propiedad de la fórmula
anterior a la síntesis, pero sí la misma información que la línea base de balance de carga
(F2.4). Si el Pipeline añade un indicador de dato faltante, el modelo la usa directamente.

## 3. Radios sustitutos

Cuando el ion no tiene radio de Shannon con NC 12 (sitio A) o 6 (sitio B), se usa el NC
disponible más cercano, como hace Bartel. r_A sustituto: {sub_by_label["P"]:.1%} de las P y
{sub_by_label["U"]:.1%} de las U de `train_pool`. Casos más frecuentes:

{pd.DataFrame(top_sub_rows).to_markdown(index=False)}

Son sobre todo lantánidos 3+ (NC 9–10), Bi³⁺ y Tl³⁺ (NC 8), Mg²⁺ y Zn²⁺ (NC 8) y Sn²⁺ (NC 6).
El radio sustituto subestima r_A, y con él t y τ, de forma sistemática para esos elementos.
Es un sesgo por elemento, no por etiqueta: la misma composición recibe el mismo radio sea P o
U, y la frecuencia es parecida en ambas clases.

## 4. Sitios de Bartel frente a los de la estructura

En las positivas, el A de Bartel coincide con el de la estructura en el
{site_agreement["agreement"]:.1%} de los casos. Discrepancias: {dis}.

## 5. Solapamiento con la TableS1 de Bartel (τ *in-sample*)

- Positivas de `train_pool` en la TableS1: **{overlap["P_in_TableS1"]}**; fuera:
  **{overlap["P_outside_TableS1"]}**. Esas {overlap["P_outside_TableS1"]} positivas son el
  único conjunto donde τ está fuera de muestra (sensibilidad 14 y sensibilidad 9 de H2). Con
  tan pocas, la potencia es baja.
- No etiquetadas en la TableS1: {overlap["U_in_TableS1"]} (por `exp_label` de Bartel:
  {overlap["U_in_TableS1_by_exp_label"]}). Las de `exp_label = 1` son las perovskitas
  experimentales de Bartel que MP 2026 solo tiene como teóricas (bitácora, 2026-11-13). Para τ
  también son *in-sample*: la evaluación "fuera de Bartel" excluye toda la TableS1, tanto las
  P como las U.
- Positivas nuestras que Bartel da como NO perovskita: {overlap["P_in_TableS1_non_perovskite_for_bartel"]} (`flag_bartel_conflict`).

## 6. Dos definiciones de balance de carga

| | P que fallan | U que fallan |
|---|---|---|
| `compound possible` de matminer (estados **comunes** de pymatgen) | {cbp["common_states_fail"]} de {cbp["n"]} | {cbu["common_states_fail"]} de {cbu["n"]} |
| `flag_no_charge_balance` de F1.5 (`oxi_state_guesses`: estados **observados en ICSD**) | {cbp["icsd_states_fail"]} de {cbp["n"]} | {cbu["icsd_states_fail"]} de {cbu["n"]} |

Con los estados comunes fallan perovskitas clásicas como LaMnO₃, porque Mn³⁺ no es "común"
en pymatgen. `compound possible` entra en `pre_dft` como una feature más. La definición de la
línea base de balance de carga se decide en F2.4.

## 7. Distribución descriptiva de t, μ y τ (mediana [Q1, Q3])

{dist.assign(clase=dist["clase"].map(CLASS_NAMES)).to_markdown(index=False)}

Es solo descriptiva, no una evaluación: la de τ como clasificador (línea base) se hace en F2.4
con los folds de cada esquema.
"""
    (OUT / "informe.md").write_text(md, "utf-8")
    print(json.dumps({k: s[k] for k in ("families", "coverage", "table_s1_overlap", "charge_balance")},
                     indent=1, ensure_ascii=False))  # fmt: skip


if __name__ == "__main__":  # requisito en Windows (regla 2.2)
    main()
