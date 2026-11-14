"""F3.4: ejecución de la validación de formabilidad desde su YAML.

Dos variantes para puntuar los negativos (decisión de la autora, 2026-11-14):

- `final`: el modelo PU entrenado con TODO train_pool, con el mismo protocolo interno que F3.3
  (búsqueda de 15 candidatos y umbral a recall 0,90 en un CV interno). Ningún negativo está en
  train_pool, así que no hay fuga. Mide la formabilidad del modelo que se usaría en la práctica.
- `family`: cada negativo lo puntúa el modelo del fold de `family` que reserva la familia de
  su elemento B (extrapolación química). Con las mismas semillas y candidatos que F3.3, el
  modelo del fold es EL MISMO que en F3.3: las puntuaciones de las composiciones de train_pool
  del test deben coincidir con las de F3.3 (comprobación de coherencia en el informe).

Las reglas (τ, ventanas de t, balance de carga) se evalúan igual en P, U y negativos. τ es
*in-sample* en los negativos de la TableS1 de Bartel (se ajustó con ellos).

Salidas: results/f3_4_formability/<timestamp>_<hash>/ con raw/seed_<n>.csv, raw/rules.csv,
raw/negatives.csv (conjuntos), metrics_by_seed.csv, summary.csv y run_metadata.json; copia
del resumen en results/f3_4_formability/summary.csv y fila en results/runs_index.csv.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from perovskite_pu.evaluation.formability import (
    SCORED_SETS,
    auc_positive_vs_negative,
    formability_sets,
    positive_rate,
)

RAW_COLUMNS = ["variant", "model", "seed", "fold", "group", "composition", "set", "label",
               "score", "threshold", "pred"]  # fmt: skip


# ---------------------------------------------------------------------------
# Entradas de los negativos
# ---------------------------------------------------------------------------
def negative_feature_tables(
    compositions: Sequence[str], features_cfg: Mapping[str, Any], table_s1: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Features `pre_dft` y `pre_dft_tau` de los negativos, en el orden pedido.

    Las sintetizadas no perovskita y las ambiguas ya se calcularon en F2.1. Las que faltan
    (los polianiones de Bartel y alguna fuera de nuestro espacio) se calculan aquí con el
    mismo código (`build_pre_dft`), sin escribir nada en data/interim/.
    """
    from perovskite_pu.features import sets
    from perovskite_pu.features.pre_dft import build_pre_dft
    from perovskite_pu.utils.paths import ROOT

    comps = list(compositions)
    stored = pd.read_parquet(ROOT / sets.output_paths()["table"])
    have = [c for c in comps if c in stored.index]
    missing = [c for c in comps if c not in stored.index]
    cols_pre = sets.load_feature_matrix("pre_dft", have[:1]).columns
    cols_tau = sets.load_feature_matrix("pre_dft_tau", have[:1]).columns
    parts = [stored.loc[have]]
    if missing:
        table, _, _ = build_pre_dft(missing, features_cfg, table_s1)
        parts.append(table)
    full = pd.concat(parts).loc[comps]
    return full[cols_pre], full[cols_tau]


def negative_rule_inputs(
    compositions: Sequence[str], tau_table: pd.DataFrame, labels: pd.DataFrame
) -> pd.DataFrame:
    """Entradas de las reglas para los negativos (mismas definiciones que `baseline_inputs`)."""
    comps = list(compositions)
    flags = labels.set_index("reduced_formula")["flag_no_charge_balance"]
    no_balance = flags.reindex(comps)
    out = pd.DataFrame(index=pd.Index(comps, name="composition"))
    out["tau"] = tau_table["geom_tau"].to_numpy()
    out["t"] = tau_table["geom_t"].to_numpy()
    # Sin etiqueta de F1.5 (fuera de nuestro espacio) no hay estados de ICSD: NaN.
    balanced = (~no_balance.astype("boolean").fillna(False).astype(bool)).astype(float)
    out["charge_balanced_icsd"] = np.where(no_balance.isna(), np.nan, balanced)
    out["charge_balanced_common"] = tau_table["compound possible"].astype(float).to_numpy()
    return out


# ---------------------------------------------------------------------------
# Tareas
# ---------------------------------------------------------------------------
def _rows(variant, model, seed, fold, group, comps, kinds, labels, scores, thr):
    """Filas crudas de un bloque de puntuaciones con su umbral."""
    scores = np.asarray(scores, float)
    return pd.DataFrame({
        "variant": variant, "model": model, "seed": seed, "fold": fold, "group": group,
        "composition": list(comps), "set": list(kinds), "label": np.asarray(labels, int),
        "score": scores, "threshold": thr, "pred": (scores >= thr).astype(int),
    })  # fmt: skip


def _final_task(name, spec, cands, X_pool, s, X_neg, neg, seed, inner_cfg):
    """Variante `final`: modelo con todo train_pool; puntúa todos los negativos."""
    from perovskite_pu.training.runner import select_and_fit_pu

    scores, thr, _, _ = select_and_fit_pu(spec, cands, X_pool, s, X_neg, seed, "final", 0,
                                          inner_cfg)  # fmt: skip
    return _rows("final", name, seed, -1, "all", neg["composition"], neg["set"],
                 np.full(len(neg), -1), scores, thr)  # fmt: skip


def _family_task(name, spec, cands, X_pool, s, comps, fold, k, X_neg, neg, mask, seed,
                 inner_cfg):  # fmt: skip
    """Variante `family`: el modelo del fold puntúa su test de train_pool y sus negativos."""
    from perovskite_pu.training.runner import select_and_fit_pu

    tr, te = fold.train_idx, fold.test_idx
    X_te = np.vstack([X_pool[te], X_neg[mask]])
    scores, thr, _, _ = select_and_fit_pu(spec, cands, X_pool[tr], s[tr], X_te, seed, "family", k,
                                          inner_cfg)  # fmt: skip
    n_te = len(fold.test_idx)
    group = "+".join(fold.test_groups)
    s_te = s[fold.test_idx]
    kinds = np.where(s_te == 1, "positive", "unlabeled")
    test_comps = [comps[i] for i in fold.test_idx]
    pool_rows = _rows("family", name, seed, k, group, test_comps, kinds, s_te, scores[:n_te], thr)
    neg_lab = np.full(int(mask.sum()), -1)
    neg_rows = _rows("family", name, seed, k, group, neg.loc[mask, "composition"],
                     neg.loc[mask, "set"], neg_lab, scores[n_te:], thr)  # fmt: skip
    return pd.concat([pool_rows, neg_rows], ignore_index=True)


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def formability_metrics(raw: pd.DataFrame, principal_set: str = "principal") -> pd.DataFrame:
    """Por variante × modelo × semilla: TPR, P(ŷ=1|U), FPR por conjunto y AUC P frente a N.

    En `final` las P y U son de entrenamiento: no hay TPR ni AUC (su TPR de referencia es la
    de F3.3). En `family` y en las reglas, la AUC usa las P del test y los negativos
    `principal` puntuados por los mismos modelos (agregada sobre folds).
    """
    out = []
    for (variant, model, seed), g in raw.groupby(["variant", "model", "seed"], sort=True):
        pos, unl = g[g["set"] == "positive"], g[g["set"] == "unlabeled"]
        row = {"variant": variant, "model": model, "seed": seed,
               "n_pos": len(pos), "n_u": len(unl),
               "tpr": positive_rate(pos["pred"]), "ppr_u": positive_rate(unl["pred"])}  # fmt: skip
        for kind in SCORED_SETS:
            neg = g[g["set"] == kind]
            row[f"n_{kind}"] = len(neg)
            row[f"fpr_{kind}"] = positive_rate(neg["pred"])
        prin = g[g["set"] == principal_set]
        row["auc_principal"] = auc_positive_vs_negative(pos["score"], prin["score"])
        out.append(row)
    return pd.DataFrame(out)


def summarize_formability(metrics: pd.DataFrame) -> pd.DataFrame:
    """Media, DE e IC 95 % (t de Student sobre semillas) por variante × modelo."""
    keys = {"variant", "model", "seed"}
    rows = []
    for (variant, model), g in metrics.groupby(["variant", "model"], sort=True):
        n = len(g)
        row = {"variant": variant, "model": model, "n_seeds": n}
        for c in [c for c in metrics.columns if c not in keys]:
            v = g[c].astype(float)
            m, sd = float(v.mean()), float(v.std(ddof=1)) if n > 1 else 0.0
            half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else 0.0
            row.update({f"{c}_mean": m, f"{c}_sd": sd, f"{c}_ci95_low": m - half,
                        f"{c}_ci95_high": m + half})  # fmt: skip
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Ejecución completa
# ---------------------------------------------------------------------------
def run_formability_experiment(
    config_path: Path | str, allow_dirty: bool = False, dev_mode: bool | None = None
) -> Path:
    """Ejecuta F3.4 desde su YAML y devuelve el directorio de la corrida."""
    import json

    import yaml
    from joblib import Parallel, delayed

    from perovskite_pu.data.external import BARTEL_DIR, load_bartel2019
    from perovskite_pu.splits.schemes import assign_b_family
    from perovskite_pu.training.baselines import baseline_inputs, rule_scores
    from perovskite_pu.training.runner import _family_matrices, frozen_folds, sample_candidates
    from perovskite_pu.utils.hashing import config_hash
    from perovskite_pu.utils.paths import RESULTS_DIR, ROOT
    from perovskite_pu.utils.run_metadata import (
        append_runs_index,
        build_run_metadata,
        make_run_dir,
        write_run_metadata,
    )
    from perovskite_pu.utils.seeds import load_seed_config

    t0 = time.perf_counter()
    cfg = yaml.safe_load(Path(config_path).read_text("utf-8"))
    data_cfg = yaml.safe_load((ROOT / cfg["data"]["config"]).read_text("utf-8"))
    # Modelos y reglas se toman de sus experimentos de origen y se guardan resueltos: el hash
    # cambia si cambian allí, aunque este YAML no cambie.
    src_models = yaml.safe_load((ROOT / cfg["model_source"]).read_text("utf-8"))
    src_rules = yaml.safe_load((ROOT / cfg["rules_source"]).read_text("utf-8"))
    resolved = {
        "models": {m: src_models["models"][m] for m in cfg["models"]},
        "inner_cv": src_models["inner_cv"],
        "rules": {r: src_rules["baselines"][r] for r in cfg["rules"]},
    }
    dev = cfg["execution"].get("dev_mode", False) if dev_mode is None else dev_mode
    seeds = load_seed_config().active_seeds(dev)
    n_jobs = int(cfg["execution"].get("n_jobs", 1))
    manifest = json.loads((ROOT / "data" / "MANIFEST.json").read_text("utf-8"))
    full_cfg = {**cfg, "resolved": resolved}
    meta = build_run_metadata(full_cfg, seeds, n_jobs=n_jobs, allow_dirty=allow_dirty,
                              mp_database_version=manifest.get("mp_database_version"))  # fmt: skip
    meta["config_hash"] = config_hash({k: v for k, v in full_cfg.items() if k != "execution"})
    meta["dev_mode"] = bool(dev)
    meta["status"] = "running"
    run_dir = make_run_dir(cfg["name"], meta["config_hash"])
    write_run_metadata(dict(meta), run_dir)

    # Datos: train_pool, conjuntos de negativos y sus features.
    pool = pd.read_parquet(ROOT / cfg["data"]["input"])
    comps = pool["composition"].tolist()
    s = pool["label"].to_numpy(int)
    labels = pd.read_parquet(ROOT / cfg["data"]["labels"])
    table_s1 = load_bartel2019(BARTEL_DIR / data_cfg["external_sources"]["bartel2019"]["table"])
    reserved = json.loads((ROOT / data_cfg["holdout"]["hashes_path"]).read_text("utf-8"))
    reserved_b = reserved["holdout"]["reserved_b"]
    neg_all = formability_sets(labels, table_s1, reserved_b, comps)
    neg_all.to_csv(run_dir / "raw" / "negatives.csv", index=False, lineterminator="\n")
    neg = neg_all[neg_all["set"].isin(cfg["negatives"]["sets"])].reset_index(drop=True)
    X_neg_df, tau_neg = negative_feature_tables(neg["composition"], data_cfg["features"],
                                                table_s1)  # fmt: skip
    X_neg = X_neg_df.to_numpy(float)
    neg["family"] = assign_b_family(neg["B"], data_cfg["b_families"]).to_numpy()
    folds_table = pd.read_parquet(ROOT / cfg["data"]["folds"])
    inner_cfg = resolved["inner_cv"]
    t_load = time.perf_counter()

    # Tareas PU: una `final` por semilla y una por fold de `family` y semilla.
    tasks = []
    for name, spec in resolved["models"].items():
        X_pool, ok = _family_matrices({name: spec}, comps)[spec["family"]]
        if not ok.all():
            raise ValueError(f"{name}: F3.4 solo admite familias definidas en todo train_pool.")
        for seed in seeds:
            cands = sample_candidates(spec["search"], seed)  # los mismos que en F3.3
            if "final" in cfg["variants"]:
                tasks.append(delayed(_final_task)(name, spec, cands, X_pool, s, X_neg, neg, seed,
                                                  inner_cfg))  # fmt: skip
            if "family" in cfg["variants"]:
                folds = frozen_folds(folds_table, comps, "family", seed)
                covered = np.zeros(len(neg), bool)
                for k, fold in enumerate(folds):
                    mask = neg["family"].isin(fold.test_groups).to_numpy()
                    covered |= mask
                    tasks.append(delayed(_family_task)(name, spec, cands, X_pool, s, comps,
                                                       fold, k, X_neg, neg, mask, seed,
                                                       inner_cfg))  # fmt: skip
                # Cada negativo debe caer en el fold de su familia de B (y solo en uno).
                if not covered.all():
                    lost = neg.loc[~covered, "composition"].tolist()[:5]
                    raise ValueError(f"Negativos sin fold de su familia en `family`: {lost}")
    parts = Parallel(n_jobs=n_jobs, backend="loky")(tasks)
    raw = pd.concat(parts, ignore_index=True)
    t_run = time.perf_counter()

    # Reglas: deterministas, en todas las P y U de train_pool y en los negativos.
    pool_in = baseline_inputs(comps)
    neg_in = negative_rule_inputs(neg["composition"], tau_neg, labels)
    rule_rows = []
    for rname, rspec in resolved["rules"].items():
        x = np.r_[pool_in[rspec["input"]].to_numpy(float), neg_in[rspec["input"]].to_numpy(float)]
        sc, thr = rule_scores(x, rspec["rule"])
        defined = ~np.isnan(x)  # la regla solo se evalúa donde su entrada existe
        kinds = np.r_[np.where(s == 1, "positive", "unlabeled"), neg["set"].to_numpy()]
        labs = np.r_[s, np.full(len(neg), -1)]
        allc = np.r_[np.array(comps), neg["composition"].to_numpy()]
        rule_rows.append(_rows("rule", rname, -1, -1, "all", allc[defined], kinds[defined],
                               labs[defined], sc[defined], thr))  # fmt: skip
    rules = pd.concat(rule_rows, ignore_index=True)

    for seed, g in raw.groupby("seed"):
        g.to_csv(run_dir / "raw" / f"seed_{seed}.csv", index=False, lineterminator="\n")
    rules.to_csv(run_dir / "raw" / "rules.csv", index=False, lineterminator="\n")
    neg.to_csv(run_dir / "raw" / "negatives_scored.csv", index=False, lineterminator="\n")
    metrics = formability_metrics(pd.concat([raw, rules], ignore_index=True),
                                  cfg["negatives"]["principal_set"])  # fmt: skip
    summary = summarize_formability(metrics)
    metrics.to_csv(run_dir / "metrics_by_seed.csv", index=False, lineterminator="\n")
    summary.to_csv(run_dir / "summary.csv", index=False, lineterminator="\n")
    summary.to_csv(RESULTS_DIR / cfg["name"] / "summary.csv", index=False, lineterminator="\n")

    meta["status"] = "completed"
    meta["n_tasks"] = len(tasks)
    meta["counts"] = neg_all["set"].value_counts().to_dict()
    meta["timings"] = {"load_s": round(t_load - t0, 3), "run_s": round(t_run - t_load, 3),
                       "total_s": round(time.perf_counter() - t0, 3)}  # fmt: skip
    write_run_metadata(meta, run_dir)
    append_runs_index(RESULTS_DIR / "runs_index.csv", cfg["name"], run_dir, meta)
    return run_dir
