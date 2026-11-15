"""Informe de F3.4: validación de formabilidad (no de sintetizabilidad).

Lee la última corrida de `f3_4_formability` en `results/runs_index.csv` (su `summary.csv` y
su `raw/`) y, para la comprobación de coherencia, el `raw/` de la última corrida de `f3_3_pu`.
Escribe `results/f3_4_formability/informe.md`. No ejecuta modelos: primero

    uv run python scripts/run_formability.py configs/experiments/f3_4_formability.yaml

Uso:
    uv run python scripts/report_f3_4.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import yaml

from perovskite_pu.utils.paths import RESULTS_DIR, ROOT
from perovskite_pu.utils.run_metadata import git_commit, git_is_dirty

EXP = "f3_4_formability"
PU_EXP = "f3_3_pu"
NAMES = {
    "pu_pre_dft": "PU pre_dft",
    "tau_bartel": "τ < 4,18 (Bartel)",
    "t_window_bartel": "t en [0,825; 1,059] (Bartel)",
    "t_window_classic": "t en [0,8; 1,0] (clásica)",
    "charge_balance_icsd": "Balance de carga (ICSD)",
    "charge_balance_common": "Balance de carga (comunes)",
}
SETS = {
    "principal": "sintetizadas no perovskita",
    "polyanion": "polianiones (Bartel)",
    "ambiguous": "ambiguas (Bartel)",
    "conflict_positive": "conflicto: P para nosotros",
    "conflict_unlabeled": "conflicto: U para nosotros",
    "excluded_reserved_b": "B reservado (no se puntúa)",
}


def last_run(index: pd.DataFrame, exp: str) -> pd.Series | None:
    """Última corrida de un experimento en `runs_index.csv` (None si no hay)."""
    rows = index[index["experiment"] == exp]
    return rows.iloc[-1] if len(rows) else None


def cell(row: pd.Series, col: str) -> str:
    """'media [IC 95 %]'; un solo número si no varía entre semillas (reglas)."""
    m, lo, hi = row[f"{col}_mean"], row[f"{col}_ci95_low"], row[f"{col}_ci95_high"]
    if pd.isna(m):
        return "—"
    return f"{m:.3f}" if abs(hi - lo) < 5e-4 else f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def main() -> None:
    index = pd.read_csv(RESULTS_DIR / "runs_index.csv")
    run = last_run(index, EXP)
    if run is None:
        raise SystemExit("No hay corridas de f3_4_formability en runs_index.csv.")
    run_dir = ROOT / run["run_dir"]
    meta = json.loads((run_dir / "run_metadata.json").read_text("utf-8"))
    cfg = meta["config"]
    summary = pd.read_csv(run_dir / "summary.csv")
    raw = pd.concat([pd.read_csv(p) for p in sorted((run_dir / "raw").glob("seed_*.csv"))],
                    ignore_index=True)  # fmt: skip
    rules = pd.read_csv(run_dir / "raw" / "rules.csv")
    negs = pd.read_csv(run_dir / "raw" / "negatives.csv")
    seeds = sorted(int(x) for x in raw["seed"].unique())

    # 1. Conjuntos de negativos y su procedencia.
    counts = (negs.groupby(["set", "source"]).size().unstack(fill_value=0)
              .reindex(list(SETS)).fillna(0).astype(int))  # fmt: skip
    counts.index = [SETS[k] for k in counts.index]

    # 2. Tabla principal: TPR, P(ŷ=1|U), FPR por conjunto y AUC, por variante y modelo/regla.
    def table(variant: str, models: list[str]) -> str:
        rows = []
        for m in models:
            hit = summary[(summary["variant"] == variant) & (summary["model"] == m)]
            if not len(hit):
                continue
            h = hit.iloc[0]
            rows.append({"modelo": NAMES.get(m, m), "TPR (P)": cell(h, "tpr"),
                         "P(ŷ=1|U)": cell(h, "ppr_u"),
                         "FPR sint. no perov.": cell(h, "fpr_principal"),
                         "FPR polianiones": cell(h, "fpr_polyanion"),
                         "FPR ambiguas": cell(h, "fpr_ambiguous"),
                         "AUC P vs sint. no perov.": cell(h, "auc_principal")})  # fmt: skip
        return pd.DataFrame(rows).to_markdown(index=False)

    pu_models = list(cfg["models"])
    rule_names = list(cfg["rules"])

    # 3. τ in-sample: FPR de las reglas y del PU en las negativas de la TableS1 frente al resto.
    in_s1 = set(negs.loc[negs["in_bartel2019"].astype(bool), "composition"])
    split_rows = []
    principal = negs.loc[negs["set"] == "principal", "composition"]
    for name, df in [*((m, raw[(raw["variant"] == "final") & (raw["model"] == m)])
                       for m in pu_models),
                     *((r, rules[rules["model"] == r]) for r in rule_names)]:  # fmt: skip
        d = df[df["composition"].isin(principal)]
        a, b = d[d["composition"].isin(in_s1)], d[~d["composition"].isin(in_s1)]
        split_rows.append({"modelo": NAMES.get(name, name),
                           "FPR en la TableS1 (τ in-sample)": f"{a['pred'].mean():.3f} (n={a['composition'].nunique()})",
                           "FPR fuera de la TableS1": f"{b['pred'].mean():.3f} (n={b['composition'].nunique()})"})  # fmt: skip

    # 4. Por familia de B (variante family): FPR en sintetizadas no perovskita y TPR del fold.
    fam = raw[raw["variant"] == "family"]
    fam_rows = []
    for (m, grp), g in fam.groupby(["model", "group"], sort=True):
        pos, neg = g[g["set"] == "positive"], g[g["set"] == "principal"]
        fam_rows.append({"modelo": NAMES.get(m, m), "familia de B": grp,
                         "P (test)": pos["composition"].nunique(),
                         "sint. no perov.": neg["composition"].nunique(),
                         "TPR": f"{pos['pred'].mean():.3f}",
                         "FPR": f"{neg['pred'].mean():.3f}" if len(neg) else "—"})  # fmt: skip

    # 5. Negativos que el modelo final confunde con perovskita en ≥ la mitad de las semillas.
    fin = raw[(raw["variant"] == "final") & (raw["set"] == "principal")]
    freq = fin.groupby("composition")["pred"].mean().rename("fracción de semillas")
    tau = rules[rules["model"] == "tau_bartel"].set_index("composition")["pred"]
    confused = freq[freq >= 0.5].sort_values(ascending=False).to_frame()
    confused["τ predice perovskita"] = tau.reindex(confused.index).map({1: "sí", 0: "no"})
    confused["en la TableS1"] = [("sí" if c in in_s1 else "no") for c in confused.index]

    # 6. Coherencia con F3.3: los modelos de los folds de `family` son los mismos.
    pu_run = last_run(index, PU_EXP)
    coherence = "Sin corrida de F3.3 en `runs_index.csv`: comprobación pendiente."
    if pu_run is not None:
        f33 = pd.concat([pd.read_csv(p) for p in
                         sorted((ROOT / pu_run["run_dir"] / "raw").glob("seed_*.csv"))],
                        ignore_index=True)  # fmt: skip
        f33 = f33[(f33["scheme"] == "family") & f33["seed"].isin(seeds)]
        mine = fam[fam["set"].isin(["positive", "unlabeled"])]
        merged = mine.merge(f33, left_on=["model", "seed", "composition"],
                            right_on=["baseline", "seed", "composition"], suffixes=("", "_f33"))  # fmt: skip
        if len(merged):
            diff = float(np.abs(merged["score"] - merged["score_f33"]).max())
            same_thr = bool(np.allclose(merged["threshold"], merged["threshold_f33"]))
            coherence = (f"{len(merged)} puntuaciones de train_pool comparadas con F3.3 "
                         f"(`{pu_run['run_dir']}`): diferencia máxima {diff:.2e}; umbrales "
                         f"{'idénticos' if same_thr else 'DISTINTOS'}.")  # fmt: skip

    md = f"""# Informe F3.4: validación de formabilidad

> Generado por `scripts/report_f3_4.py` (commit `{git_commit()[:7]}`{", árbol sucio" if git_is_dirty() else ""}) a partir de
> `{run["run_dir"]}` (corrida con el commit `{str(run["git_commit"])[:7]}`{", árbol sucio" if bool(run["git_dirty"]) else ""}; semillas {seeds}; {run["total_s"] / 60:.0f} min).
> No editar a mano. Configuración: `configs/experiments/{EXP}.yaml`. Solo `train_pool`.

## 1. Qué se mide

- **Formabilidad, no sintetizabilidad:** los negativos son óxidos ABO₃ que se han sintetizado
  y no son perovskita. Un buen modelo de "sintetizable como perovskita" debería puntuarlos
  por debajo de las perovskitas conocidas. Aquí hay negativos verdaderos: la FPR y la ROC-AUC
  entre P y negativos son métricas reales, no cotas inferiores.
- **Variante `final`:** el modelo PU entrenado con todo train_pool con el protocolo de F3.3
  (umbral a recall 0,90 en el CV interno). Sus P y U son de entrenamiento: su TPR de
  referencia es la de F3.3.
- **Variante `family`:** cada negativo lo puntúa el modelo del fold de `family` que reserva
  la familia de su elemento B (extrapolación). TPR y AUC usan las P del test de esos folds.
- **Reglas:** τ, ventanas de t y balance de carga, con sus umbrales externos. **τ es
  *in-sample*** en los negativos de la TableS1 (Bartel et al. la ajustaron con ellos).
- Media de {len(seeds)} semillas e IC 95 % (t de Student); las reglas no varían entre semillas.

Conjuntos (procedencia: nuestra etiquetación, Bartel o ambas):

{counts.to_markdown()}

## 2. Modelo final (todo train_pool)

{table("final", pu_models)}

## 3. Extrapolación: folds de `family` y reglas

{table("family", pu_models)}

Reglas (en todas las P y U de train_pool y en todos los negativos donde están definidas):

{table("rule", rule_names)}

## 4. τ *in-sample*: negativos de la TableS1 frente al resto (sintetizadas no perovskita)

{pd.DataFrame(split_rows).to_markdown(index=False)}

## 5. Por familia de B (variante `family`)

{pd.DataFrame(fam_rows).to_markdown(index=False)}

## 6. Sintetizadas no perovskita que el modelo final predice como perovskita

En al menos la mitad de las semillas:

{confused.to_markdown() if len(confused) else "Ninguna."}

## 7. Coherencia con F3.3

{coherence}
"""
    out = RESULTS_DIR / EXP / "informe.md"
    out.write_text(md, "utf-8")
    (RESULTS_DIR / EXP / "config_used.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), "utf-8"
    )
    print(md)


if __name__ == "__main__":  # requisito en Windows (regla 2.2)
    main()
