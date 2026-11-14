"""Informe de F2.3: auditoría de features, incluida la circularidad de E_hull (regla 2.1.8).

Cruza `configs/feature_audit.yaml` con las familias calculadas en F2.1–F2.2 y con
`train_pool`; el holdout no se toca. El cribado univariante es DESCRIPTIVO: no elige features.

Escribe en `results/f2_3_auditoria/`: summary.json, univariate_screen.csv, informe.md.

Uso:
    uv run python scripts/report_f2_3.py
"""

from __future__ import annotations

import json
import re

import pandas as pd
import yaml

from perovskite_pu.features import sets
from perovskite_pu.features.audit import (
    audited_columns,
    check_family_columns,
    load_audit_config,
    suspicious,
    univariate_auc,
)
from perovskite_pu.utils.paths import (
    CONFIGS_DIR,
    INTERIM_DIR,
    PROCESSED_DIR,
    RESULTS_DIR,
    ROOT,
    TRAIN_POOL_FILENAME,
)
from perovskite_pu.utils.run_metadata import git_commit, git_is_dirty

OUT = RESULTS_DIR / "f2_3_auditoria"
STAGE = {"pre_synthesis": "previa a la síntesis", "post_dft": "DFT (MP)"}
DEP = {"none": "ninguna", "in_sample": "**in-sample**", "indirect": "**indirecta**"}


def main() -> None:
    audit = load_audit_config()
    out_cfg = yaml.safe_load((CONFIGS_DIR / "data.yaml").read_text("utf-8"))["features"]["output"]
    pre = json.loads((ROOT / out_cfg["columns"]).read_text("utf-8"))
    post = json.loads((ROOT / out_cfg["post_columns"]).read_text("utf-8"))
    families = {**pre["families"], **post["families"]}
    col2group = audited_columns(audit["groups"], pre["groups"])
    tp = pd.read_parquet(PROCESSED_DIR / TRAIN_POOL_FILENAME)

    # 1. Grupos auditados y familias que los usan.
    group_rows = []
    for g, spec in audit["groups"].items():
        cols = [c for c, gg in col2group.items() if gg == g]
        fams = [f for f, fc in families.items() if set(cols) & set(fc)]
        group_rows.append({
            "grupo": g, "columnas": len(cols), "etapa": STAGE[spec["stage"]],
            "dependencia de la etiqueta": DEP[spec["label_dependence"]],
            "familias": ", ".join(fams) or "—",
        })  # fmt: skip

    # 2. Comprobaciones automáticas (las mismas que el test de fuga dependiente de datos).
    problems = check_family_columns(families, audit, pre["groups"])

    # 3. Columnas que existen en los datos y NUNCA pueden ser features.
    patterns = [re.compile(p) for p in audit["forbidden_patterns"]]
    sources = {
        "train_pool": list(tp.columns),
        "f1_5_entries": list(pd.read_parquet(INTERIM_DIR / "f1_5_entries.parquet").columns),
        "features_pre_dft_meta": list(pd.read_parquet(ROOT / out_cfg["metadata"]).columns),
        "features_post_dft_meta": list(pd.read_parquet(ROOT / out_cfg["post_metadata"]).columns),
    }
    blocked = {k: sorted(c for c in v if any(p.search(c) for p in patterns))
               for k, v in sources.items()}  # fmt: skip

    # 4. Cribado univariante (descriptivo): pre_dft_tau en train_pool; termo en elegibles.
    y = tp["label"].to_numpy()
    screen_pre = univariate_auc(sets.load_feature_matrix("pre_dft_tau", tp["composition"]), y)
    ok = sets.eligible("post_dft", tp["composition"]).to_numpy()
    Xp = sets.load_feature_matrix("post_dft", tp["composition"][ok])
    thermo_cols = [c for c in Xp.columns if c.startswith("thermo_")]
    screen_post = univariate_auc(Xp[thermo_cols], y[ok])
    screen = pd.concat([screen_pre.assign(family="pre_dft_tau"),
                        screen_post.assign(family="post_dft (elegibles)")], ignore_index=True)  # fmt: skip
    thr = audit["screen"]["suspicious_auc"]
    flagged = suspicious(screen, thr)
    best_pre = screen_pre[~screen_pre["feature"].isin(["geom_tau"])].iloc[0]
    tau_row = screen_pre[screen_pre["feature"] == "geom_tau"].iloc[0]
    ehull_row = screen_post[screen_post["feature"] == "thermo_ehull"].iloc[0]

    # AUC de la ausencia de t (indicador de dato faltante) como separador.
    t_missing = sets.load_feature_matrix("pre_dft", tp["composition"])["geom_t"].isna()
    miss_auc = univariate_auc(t_missing.astype(float).to_frame("t_missing"), y).iloc[0]["auc"]

    # 5. Circularidad de E_hull: P sobre el hull y entrada experimental.
    meta_post = pd.read_parquet(ROOT / out_cfg["post_metadata"])
    comps_ok = tp["composition"][ok]
    on_hull = (Xp["thermo_ehull"] < 1e-6).groupby(y[ok]).mean()
    exp_frac = meta_post.loc[comps_ok, "experimental_gga"].astype(bool).groupby(y[ok]).mean()

    s = {
        "code_commit": git_commit(), "code_dirty": git_is_dirty(),
        "problems": problems, "groups": group_rows, "blocked_columns": blocked,
        "suspicious_threshold": thr, "suspicious": flagged.to_dict("records"),
        "best_pre_dft_feature": best_pre.to_dict(), "tau": tau_row.to_dict(),
        "ehull": ehull_row.to_dict(), "t_missing_auc": float(miss_auc),
        "on_hull_frac": {"P": float(on_hull.get(1)), "U": float(on_hull.get(0))},
        "chosen_entry_experimental_frac": {"P": float(exp_frac.get(1)), "U": float(exp_frac.get(0))},
    }  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(
        json.dumps(s, indent=2, ensure_ascii=False, default=float) + "\n", "utf-8"
    )
    screen.to_csv(OUT / "univariate_screen.csv", index=False, lineterminator="\n")

    notes = "\n".join(
        f"- **`{g}`.** Fuente: {spec['source']}."
        + (f" {spec['note']}" if spec.get("note") else "")
        + (f" *Mitigación:* {spec['mitigation']}" if spec.get("mitigation") else "")
        for g, spec in audit["groups"].items()
    )
    blocked_md = "\n".join(
        f"- `{k}`: {', '.join(f'`{c}`' for c in v) or '—'}" for k, v in blocked.items()
    )
    top = screen_pre.head(10)[["feature", "auc", "nan_frac_P", "nan_frac_U"]].round(3)
    md = f"""# Informe F2.3: auditoría de features

> Generado por `scripts/report_f2_3.py` (commit `{s["code_commit"][:7]}`{", árbol sucio" if s["code_dirty"] else ""}).
> No editar a mano. Especificación: `configs/feature_audit.yaml`. Cifras sobre `train_pool`
> ({len(tp)} composiciones, {int(y.sum())} P); el holdout no se toca.

## 1. Grupos de features

{pd.DataFrame(group_rows).to_markdown(index=False)}

{notes}

## 2. Comprobaciones automáticas

Toda columna de toda familia pertenece a un grupo auditado (lista blanca), ninguna coincide con
un patrón prohibido, y `pre_dft` solo lleva grupos previos a la síntesis sin dependencia de la
etiqueta. **Resultado: {"sin problemas" if not problems else f"{len(problems)} problemas: " + "; ".join(problems)}.** Las mismas comprobaciones están en
`tests/test_leakage.py`: la de configuración siempre activa y la de datos con `REQUIRE_DATA=1`.

## 3. Columnas bloqueadas

Existen en los datos, pero nunca pueden ser features (metadatos, procedencia o información
derivada de la etiqueta). `features.sets` solo devuelve las columnas de cada familia.

{blocked_md}

## 4. Cribado univariante (descriptivo; no selecciona features)

AUC de cada columna para separar P de U, con U como negativo (cota inferior) y sin NaN. Se
marca como sospechosa si AUC ≥ {thr} o ≤ {1 - thr:.2f}. **Sospechosas: {len(flagged)}{"" if flagged.empty else " (" + ", ".join(flagged["feature"]) + ")"}.**

Diez columnas más separadoras de `pre_dft_tau`:

{top.to_markdown(index=False)}

- Mejor columna de `pre_dft` (sin τ): `{best_pre["feature"]}`, AUC {best_pre["auc"]:.3f}.
- τ (in-sample): AUC {tau_row["auc"]:.3f}; es decir, {1 - tau_row["auc"]:.3f} para "τ bajo → P".
- Ausencia de t como separador (indicador de dato faltante): AUC {s["t_missing_auc"]:.3f}, es decir,
  {1 - s["t_missing_auc"]:.3f} para "sin t → U". Es
  información legítima (sin balance de carga con las reglas de Bartel); queda pendiente para
  F3.1 si el imputador añade el indicador.

## 5. Circularidad de E_hull

- E_hull del polimorfo perovskita (elegibles): AUC {ehull_row["auc"]:.3f}, es decir,
  **{1 - ehull_row["auc"]:.3f} para "E_hull baja → P"**, una sola columna frente a {max(best_pre["auc"], 1 - best_pre["auc"]):.3f} de la mejor de `pre_dft`.
- Sobre el hull (E_hull = 0): {s["on_hull_frac"]["P"]:.1%} de las P y {s["on_hull_frac"]["U"]:.1%} de las U elegibles.
- Entrada elegida experimental: {s["chosen_entry_experimental_frac"]["P"]:.1%} de las P y {s["chosen_entry_experimental_frac"]["U"]:.1%} de las U.

Mecanismo: el hull de MP lo definen en parte las fases de ICSD, y en las P la E_hull es la de
su estructura experimental relajada, mientras que en las U es la de un prototipo hipotético,
a menudo sin relajar sus inclinaciones. Una E_hull baja dice en buena parte "esta fase está en
ICSD". Por eso `post_dft` se presenta solo como **cota superior informada por el hull**, junto
a la línea base de solo E_hull (F2.4), y nunca como evidencia de sintetizabilidad.
"""
    (OUT / "informe.md").write_text(md, "utf-8")
    print(json.dumps({k: s[k] for k in ("problems", "suspicious", "t_missing_auc", "on_hull_frac")},
                     indent=1, ensure_ascii=False, default=float))  # fmt: skip


if __name__ == "__main__":  # requisito en Windows (regla 2.2)
    main()
