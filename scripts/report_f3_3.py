"""Informe de F3.3: bagging PU (pre_dft y post_dft) frente a las líneas base de F2.4.

Lee la última corrida de `f3_3_pu` en `results/runs_index.csv` (su `raw/`, que no se
versiona, y su `summary.csv`) y la última de `f2_4_baselines` (su `metrics_by_seed.csv`).
Escribe `results/f3_3_pu/informe.md` y `results/f3_3_pu/summary_same_subset.csv`. No ejecuta
modelos: primero

    uv run poe experiment configs/experiments/f3_3_pu.yaml

Comparación justa con las líneas base: cada línea base se evalúa en su subconjunto (p. ej. τ
solo donde está definida). Para comparar, las métricas del modelo PU se recalculan sobre las
mismas composiciones y con las mismas semillas, con `seed_metrics` (la P(ŷ=1) fija se aplica
dentro de cada fold del subconjunto, igual que en la línea base). El umbral del modelo PU no
cambia: es el calibrado en su bucle interno con todo su train. Todo es descriptivo: las
pruebas formales (Nadeau–Bengio, bootstrap jerárquico por grupo) son de F4.

Uso:
    uv run python scripts/report_f3_3.py [--run-dir DIR] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from perovskite_pu.evaluation.metrics import expected_hits_at_ppr
from perovskite_pu.training.runner import seed_metrics, summarize
from perovskite_pu.utils.paths import RESULTS_DIR, ROOT
from perovskite_pu.utils.run_metadata import git_commit, git_is_dirty

EXP = "f3_3_pu"
BASE_EXP = "f2_4_baselines"
SCHEMES = ["random", "group_system", "loeo_A", "loeo_B", "family", "cluster", "temporal"]
CONFIRMATORY = ["loeo_A", "loeo_B", "family", "cluster"]
NAMES = {
    "pu_pre_dft": "PU pre_dft",
    "pu_post_dft": "PU post_dft",
    "tau_bartel": "τ < 4,18 (Bartel)",
    "t_window_bartel": "t en [0,825; 1,059] (Bartel)",
    "ehull_0p05": "E_hull ≤ 0,05",
    "ehull_model": "Modelo solo E_hull",
    "symmetry_model": "Modelo solo simetría",
    "charge_balance_icsd": "Balance de carga (ICSD)",
    "trivial_random": "Azar",
}
# Bloques de comparación: (modelo PU, subconjunto, líneas base evaluadas en ese subconjunto).
BLOCKS = [
    ("pu_pre_dft", "all", ["charge_balance_icsd", "trivial_random"]),
    ("pu_pre_dft", "tau_defined", ["tau_bartel"]),
    ("pu_pre_dft", "t_defined", ["t_window_bartel"]),
    ("pu_pre_dft", "post_dft", []),  # misma población que post_dft: ¿aporta el hull?
    ("pu_post_dft", "post_dft", ["ehull_model", "ehull_0p05", "symmetry_model"]),
]
SUBSET_NAMES = {
    "all": "todas (train_pool)",
    "tau_defined": "τ definida",
    "t_defined": "t definida",
    "post_dft": "elegibles post_dft",
}


def last_run(index: pd.DataFrame, exp: str) -> pd.Series:
    """Fila de la última corrida completada de un experimento en `runs_index.csv`."""
    return index[index["experiment"] == exp].iloc[-1]


def cell(row: pd.Series, metric: str) -> str:
    """'media [IC 95 %]' con 3 decimales; un solo número si no varía entre semillas."""
    m, lo, hi = row[f"{metric}_mean"], row[f"{metric}_ci95_low"], row[f"{metric}_ci95_high"]
    if pd.isna(m):
        return "—"
    return f"{m:.3f}" if abs(hi - lo) < 5e-4 else f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def scheme_table(summary: pd.DataFrame, metric: str, models: list[str]) -> str:
    """Tabla modelo × esquema de una métrica."""
    rows = []
    for b in models:
        r = {"modelo": NAMES.get(b, b)}
        for sch in SCHEMES:
            hit = summary[(summary["baseline"] == b) & (summary["scheme"] == sch)]
            r[sch] = cell(hit.iloc[0], metric) if len(hit) else "—"
        rows.append(r)
    return pd.DataFrame(rows).to_markdown(index=False)


def same_subset_summary(raw: pd.DataFrame, comps: pd.DataFrame, base_metrics: pd.DataFrame,
                        fixed_ppr: list[float]) -> pd.DataFrame:  # fmt: skip
    """Resumen de cada bloque de comparación (modelo PU restringido + líneas base)."""
    seeds = sorted(int(x) for x in raw["seed"].unique())
    out = []
    for model, subset, baselines in BLOCKS:
        keep = set(comps.loc[comps[subset].astype(bool), "composition"])
        sub = raw[(raw["baseline"] == model) & raw["composition"].isin(keep)]
        parts = [summarize(seed_metrics(sub, fixed_ppr))]
        # Líneas base: mismas semillas que la corrida PU (el resto no entra en la media).
        bm = base_metrics[base_metrics["baseline"].isin(baselines)
                          & base_metrics["seed"].isin(seeds)]  # fmt: skip
        if len(bm):
            parts.append(summarize(bm))
        block = pd.concat(parts, ignore_index=True)
        block.insert(0, "subset", subset)
        block.insert(0, "block_model", model)
        out.append(block)
    return pd.concat(out, ignore_index=True)


def block_table(same: pd.DataFrame, model: str, subset: str, schemes: list[str],
                ppr_key: str) -> str:  # fmt: skip
    """Tabla de un bloque: filas modelo, columnas esquema × (TPR, P(ŷ=1|U), TPR a P fija)."""
    blk = same[(same["block_model"] == model) & (same["subset"] == subset)]
    rows = []
    for b in blk["baseline"].unique():
        r = {"modelo": NAMES.get(b, b)}
        for sch in schemes:
            hit = blk[(blk["baseline"] == b) & (blk["scheme"] == sch)]
            if not len(hit):
                continue
            h = hit.iloc[0]
            r[f"{sch} TPR"] = f"{h['tpr_mean']:.3f}"
            r[f"{sch} P(ŷ=1|U)"] = f"{h['ppr_u_mean']:.3f}"
            r[f"{sch} TPR@{ppr_key}"] = f"{h[f'tpr_at_ppr_{ppr_key}_mean']:.3f}"
        rows.append(r)
    return pd.DataFrame(rows).to_markdown(index=False)


def ppr_ceiling(raw: pd.DataFrame, model: str, scheme: str, ppr: float) -> float:
    """TPR máxima alcanzable a P(ŷ=1) fija: Σ_f min(P_f, ppr·n_f) / Σ_f P_f (media de semillas).

    Depende solo de cuántas positivas tiene cada fold, no del modelo: si un fold reservado es
    rico en positivas, ni un clasificador perfecto llega a TPR 1 con pocas predicciones.
    """
    g = raw[(raw["baseline"] == model) & (raw["scheme"] == scheme)]
    vals = []
    for _, gs in g.groupby("seed"):
        per_fold = gs.groupby("fold")["label"].agg(["sum", "size"])
        cap = per_fold["sum"].clip(upper=ppr * per_fold["size"]).sum()
        vals.append(cap / per_fold["sum"].sum())
    return float(sum(vals) / len(vals)) if vals else float("nan")


def h2_descriptive(raw: pd.DataFrame, tau: pd.DataFrame, exclude: set[str]) -> pd.DataFrame:
    """Contraste de H2 (descriptivo): TPR del PU a la P(ŷ=1) que da τ en cada fold.

    Sigue el preregistro §7.2: en cada fold externo, sobre las composiciones con τ definido
    (menos `exclude`, para la sensibilidad sin los compuestos de Bartel), se toma la P(ŷ=1)
    de τ < 4,18 y el modelo PU se umbraliza para igualarla (empates repartidos). Devuelve,
    por esquema y semilla, la TPR de cada uno y Δ = TPR_PU − TPR_τ micro-agregados.
    """
    out = []
    pu_all = raw[raw["baseline"] == "pu_pre_dft"]
    for (sch, seed), pu in pu_all.groupby(["scheme", "seed"], sort=True):
        # Folds de la misma semilla: en `random` y `cluster` cambian con ella.
        t_sch = tau[(tau["scheme"] == sch) & (tau["seed"] == seed)]
        hits_pu = hits_tau = n_pos = 0.0
        for fold, tf in t_sch.groupby("fold"):
            tf = tf[~tf["composition"].isin(exclude)]
            if tf.empty:  # fold sin composiciones con τ (o todas de Bartel en la sensibilidad)
                continue
            pf = pu[(pu["fold"] == fold) & pu["composition"].isin(set(tf["composition"]))]
            # Mismas composiciones en los dos modelos (folds congelados idénticos).
            if len(pf) != len(tf):
                raise RuntimeError(f"Folds distintos en {sch}/{fold}: {len(pf)} ≠ {len(tf)}")
            ppr_tau = float(tf["pred"].mean())
            hits_pu += expected_hits_at_ppr(pf["label"].to_numpy(int),
                                            pf["score"].to_numpy(float), ppr_tau)  # fmt: skip
            hits_tau += float(tf.loc[tf["label"] == 1, "pred"].sum())
            n_pos += float(tf["label"].sum())
        out.append({"scheme": sch, "seed": seed, "n_pos": int(n_pos),
                    "tpr_pu": hits_pu / n_pos, "tpr_tau": hits_tau / n_pos,
                    "delta": (hits_pu - hits_tau) / n_pos})  # fmt: skip
    return pd.DataFrame(out)


def h2_table(h2: pd.DataFrame) -> str:
    """Media (mín.–máx. entre semillas) de la TPR del PU, de τ y de Δ por esquema."""
    rows = []
    for sch in ["random", *CONFIRMATORY]:
        g = h2[h2["scheme"] == sch]
        if not len(g):
            continue
        rows.append(
            {
                "esquema": sch,
                "P": int(g["n_pos"].iloc[0]),
                "TPR τ": f"{g['tpr_tau'].mean():.3f}",
                "TPR PU (misma P(ŷ=1))": f"{g['tpr_pu'].mean():.3f}",
                "Δ (mín.–máx.)": f"{g['delta'].mean():+.3f} "
                f"({g['delta'].min():+.3f} a {g['delta'].max():+.3f})",
            }
        )
    return pd.DataFrame(rows).to_markdown(index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="corrida de f3_3_pu (por defecto, la última de runs_index.csv)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=RESULTS_DIR / EXP,
        help="dónde escribir informe.md y los CSV derivados",
    )
    args = parser.parse_args()
    index = pd.read_csv(RESULTS_DIR / "runs_index.csv")
    base_run = last_run(index, BASE_EXP)
    run_dir = (args.run_dir or ROOT / last_run(index, EXP)["run_dir"]).resolve()
    base_dir = ROOT / base_run["run_dir"]
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((run_dir / "run_metadata.json").read_text("utf-8"))
    cfg = meta["config"]
    summary = pd.read_csv(run_dir / "summary.csv")
    raw = pd.concat([pd.read_csv(p) for p in sorted((run_dir / "raw").glob("seed_*.csv"))],
                    ignore_index=True)  # fmt: skip
    search = pd.concat(
        [pd.read_csv(p) for p in sorted((run_dir / "raw").glob("search_seed_*.csv"))],
        ignore_index=True,
    )
    comps = pd.read_csv(run_dir / "raw" / "compositions.csv")
    base_metrics = pd.read_csv(base_dir / "metrics_by_seed.csv")
    fixed = cfg["evaluation"]["fixed_ppr"]
    ppr_key = f"{fixed[1]:.2f}"
    models = list(cfg["models"])
    seeds = sorted(int(x) for x in raw["seed"].unique())
    # Datos de la corrida desde su run_metadata.json (sirve aunque no esté en el índice).
    run = {
        "run_dir": run_dir.relative_to(ROOT).as_posix(),
        "git_commit": meta["git_commit"],
        "git_dirty": meta["git_dirty"],
        "total_s": meta.get("timings", {}).get("total_s", float("nan")),
        "status": meta.get("status", "desconocido"),
    }
    if run["status"] != "completed":
        print(f"AVISO: la corrida no está completada (estado: {run['status']}).")

    # H2 descriptiva con las predicciones de τ de F2.4 de las mismas semillas (raw/ no se
    # versiona; se regenera con la corrida de F2.4 en unos segundos).
    tau = pd.concat([pd.read_csv(base_dir / "raw" / f"seed_{sd}.csv") for sd in seeds],
                    ignore_index=True)  # fmt: skip
    tau = tau[tau["baseline"] == "tau_bartel"]
    bartel = set(comps.loc[comps["in_bartel2019"].astype(bool), "composition"])
    h2_main = h2_descriptive(raw, tau, exclude=set())
    h2_sens = h2_descriptive(raw, tau, exclude=bartel)
    pd.concat([h2_main.assign(analysis="principal"),
               h2_sens.assign(analysis="sin_bartel2019")]).to_csv(
        out_dir / "h2_descriptive.csv", index=False, lineterminator="\n")  # fmt: skip

    # Comparación con las líneas base sobre las mismas composiciones y semillas.
    same = same_subset_summary(raw, comps, base_metrics, fixed)
    same.to_csv(out_dir / "summary_same_subset.csv", index=False, lineterminator="\n")

    # Tamaños de evaluación (test acumulado del esquema random).
    sizes = summary[summary["scheme"] == "random"].set_index("baseline")
    size_rows = [{"modelo": NAMES[m], "familia": cfg["models"][m]["family"],
                  "papel": cfg["models"][m]["role"], "P": int(sizes.loc[m, "n_pos"]),
                  "U": int(sizes.loc[m, "n_u"])} for m in models]  # fmt: skip

    # Brecha de extrapolación: caída de TPR a P(ŷ=1) fija de random a cada esquema confirmatorio.
    def val(m: str, sch: str, metric: str) -> float:
        hit = summary[(summary["baseline"] == m) & (summary["scheme"] == sch)]
        return float(hit.iloc[0][f"{metric}_mean"]) if len(hit) else float("nan")

    gap_rows = []
    for m in models:
        r = {"modelo": NAMES[m], "TPR random": f"{val(m, 'random', 'tpr'):.3f}"}
        for sch in CONFIRMATORY:
            r[f"ΔTPR {sch}"] = f"{val(m, 'random', 'tpr') - val(m, sch, 'tpr'):+.3f}"
        gap_rows.append(r)
    gap_fixed = []
    for m in models:
        key = f"tpr_at_ppr_{ppr_key}"
        # Tres filas por modelo: TPR, techo por la fracción de positivas de cada fold y cociente.
        tpr = {sch: val(m, sch, key) for sch in ["random", *CONFIRMATORY]}
        cap = {sch: ppr_ceiling(raw, m, sch, fixed[1]) for sch in tpr}
        rows3 = {"TPR": tpr, "máximo alcanzable": cap,
                 "TPR / máximo": {sch: tpr[sch] / cap[sch] for sch in tpr}}  # fmt: skip
        for what, values in rows3.items():
            gap_fixed.append({"modelo": NAMES[m], "": what,
                              **{sch: f"{v:.3f}" for sch, v in values.items()}})  # fmt: skip

    # Aprendices elegidos en el bucle interno (fracción de folds externos).
    sel = search[search["selected"]]
    learn = (sel.groupby(["baseline", "scheme"])["learner"]
             .value_counts(normalize=True).unstack(fill_value=0.0))  # fmt: skip
    learn_rows = []
    for m in models:
        r = {"modelo": NAMES[m]}
        for sch in SCHEMES:
            if (m, sch) in learn.index:
                r[sch] = f"{learn.loc[(m, sch)].get('random_forest', 0.0):.2f}"
        learn_rows.append(r)

    # Configuraciones más elegidas. El índice de candidato cambia con la semilla (cada semilla
    # sortea las suyas), así que se cuentan por aprendiz + hiperparámetros.
    def describe(seed: int, model: str, cand: int) -> str:
        if "candidates" not in meta:  # corrida sin terminar: aún no se han guardado
            return "(sin detalle de candidatos)"
        c = meta["candidates"][str(seed)][model][cand]
        params = ", ".join(f"{k}={v}" for k, v in sorted(c["params"].items()))
        return f"{c['learner']}: {params}"

    sel = sel.assign(config=[describe(int(r.seed), r.baseline, int(r.candidate))
                             for r in sel.itertuples()])  # fmt: skip
    top = (sel.groupby(["baseline", "config"]).size().rename("n").reset_index()
           .sort_values(["baseline", "n"], ascending=[True, False])
           .groupby("baseline").head(3))  # fmt: skip
    top["baseline"] = top["baseline"].map(NAMES)

    timings = meta.get("timings", {})
    n_tasks = int(meta.get("n_tasks", 0))

    md = f"""# Informe F3.3: bagging PU ({"desarrollo, " if meta.get("dev_mode") else ""}{len(seeds)} semillas)

> Generado por `scripts/report_f3_3.py` (commit `{git_commit()[:7]}`{", árbol sucio" if git_is_dirty() else ""}) a partir de
> `{run["run_dir"]}` (corrida con el commit `{str(run["git_commit"])[:7]}`{", árbol sucio" if bool(run["git_dirty"]) else ""}; semillas {seeds};
> última ejecución {run["total_s"] / 60:.0f} min) y de las líneas base `{base_run["run_dir"]}` (mismas semillas).
> No editar a mano. Configuración: `configs/experiments/{EXP}.yaml`. Solo `train_pool`.
{"" if run["status"] == "completed" else chr(10) + f"> **Corrida no completada (estado: {run['status']}).**" + chr(10)}
## 1. Qué se mide

- **Modelo:** bagging PU (Mordelet y Vert, 2014) con {cfg["models"][models[0]]["bagging"]["n_bags"]} bolsas; el aprendiz (RF o GB) y sus
  hiperparámetros se eligen en un bucle interno estratificado de {cfg["inner_cv"]["n_splits"]} folds entre
  {cfg["models"][models[0]]["search"]["n_iter"]} candidatos por semilla, por la menor P(ŷ=1) fuera de fold al umbral de recall
  {cfg["inner_cv"]["target_recall"]:.2f} (preregistro §7.1). El umbral elegido se aplica sin cambios al fold externo.
- **TPR** micro-agregada sobre folds externos; **P(ŷ=1|U)** como control; **TPR a P(ŷ=1)
  fija** por fold con empates repartidos (§7.2). Media de las semillas e IC 95 % (t de
  Student), descriptivos.
- `pu_pre_dft` es el modelo principal; `pu_post_dft` es la cota superior informada por el hull
  (circularidad de la E_hull, regla 2.1.8).

Tamaño de evaluación (esquema `random`, test acumulado sobre folds):

{pd.DataFrame(size_rows).to_markdown(index=False)}

## 2. TPR en el punto de operación (recall nominal {f"{cfg['inner_cv']['target_recall']:.2f}".replace(".", ",")})

{scheme_table(summary, "tpr", models)}

## 3. P(ŷ = 1 | U) en el punto de operación

{scheme_table(summary, "ppr_u", models)}

## 4. TPR a P(ŷ = 1) fija

{"".join(f"P(ŷ=1) = {x:.2f}:{chr(10)}{chr(10)}{scheme_table(summary, f'tpr_at_ppr_{x:.2f}', models)}{chr(10)}{chr(10)}" for x in fixed)}
## 5. Brecha de extrapolación (H1, descriptiva)

Estadístico de H1 (§H1): ΔTPR_g = TPR_random − TPR_g en el punto de operación principal
(positivo = caída al extrapolar). Umbral de relevancia preregistrado δ₁ = 0,05. Media de
{len(seeds)} semillas, sin prueba formal (bootstrap jerárquico por grupo en F4):

{pd.DataFrame(gap_rows).to_markdown(index=False)}

La TPR en el punto de operación mezcla dos efectos: el modelo ordena peor y el umbral
calibrado en el bucle interno deja de dar recall 0,90. A P(ŷ=1) = {ppr_key} fija solo cuenta la
ordenación, pero la TPR queda acotada por la fracción de positivas de cada fold, que cambia con
el esquema; por eso se da también el máximo alcanzable y el cociente:

{pd.DataFrame(gap_fixed).to_markdown(index=False)}

## 5 bis. Frente a τ a igual P(ŷ = 1) (H2, descriptiva)

Estadístico de H2 (§7.2 y §H2): en cada fold, el PU `pre_dft` se umbraliza a la P(ŷ=1) que da
τ < 4,18 sobre las composiciones con τ definido; Δ = TPR_PU − TPR_τ. Margen de no
inferioridad δ₂ = 0,05. Análisis principal (τ *in-sample* en la TableS1, lo que favorece a τ):

{h2_table(h2_main)}

Sensibilidad sin los compuestos de Bartel et al. (2019) (τ fuera de muestra):

{h2_table(h2_sens)}

## 6. Frente a las líneas base, sobre las mismas composiciones

Cada bloque restringe el modelo PU a las composiciones donde está definida la línea base (su
umbral no cambia). Columnas: TPR en el punto de operación, P(ŷ=1|U) y TPR a P(ŷ=1) = {ppr_key}.
Medias de las semillas {seeds}; los IC están en `results/{EXP}/summary_same_subset.csv`.

{"".join(f"### {NAMES[m]} · {SUBSET_NAMES[s]}{chr(10)}{chr(10)}Esquemas `random` y confirmatorios:{chr(10)}{chr(10)}{block_table(same, m, s, ['random', *CONFIRMATORY], ppr_key)}{chr(10)}{chr(10)}" for m, s, _ in BLOCKS)}
## 7. Aprendices elegidos en el bucle interno

Fracción de folds externos en que ganó Random Forest (el resto, Gradient Boosting):

{pd.DataFrame(learn_rows).to_markdown(index=False)}

Configuraciones más elegidas (n.º de folds externos, sumando semillas y esquemas):

{top.rename(columns={"baseline": "modelo", "config": "configuración", "n": "folds"}).to_markdown(index=False)}

## 8. Coste

- {n_tasks} folds externos (semilla × esquema × fold). Última ejecución: {timings.get("run_s", float("nan")) / 60:.0f} min
  (reanudada: {timings.get("resumed")}). Si se reanudó, el tiempo de los tramos anteriores no
  está aquí: se registra en la bitácora.
"""
    out = out_dir / "informe.md"
    out.write_text(md, "utf-8")
    # Copia legible de la configuración usada, para leer el informe sin abrir el JSON.
    (out_dir / "config_used.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), "utf-8"
    )
    print(md)


if __name__ == "__main__":  # requisito en Windows (regla 2.2)
    main()
