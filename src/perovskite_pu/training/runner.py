# ATENCIÓN: este módulo solo lee train_pool y los folds congelados; nunca el conjunto de
# prueba reservado (lo verifica tests/test_leakage.py, que revisa también las cadenas).
"""Ejecutor de experimentos: YAML → folds congelados → predicciones por composición → resumen.

Flujo de `run_experiment` (regla 2.2 de CLAUDE.md):
1. Lee el YAML de `configs/experiments/` y las semillas de `configs/seeds.yaml`.
2. Se niega a correr con el árbol de git sucio, salvo `allow_dirty` (guarda el diff).
3. Para cada semilla × esquema × fold (en paralelo con joblib/loky) evalúa cada línea base:
   - reglas: puntuación y umbral externos, sin entrenamiento;
   - modelos: se ajustan SOLO con el train del fold; el umbral es el que da recall 0,90 en las
     puntuaciones fuera de fold de un bucle interno estratificado (preregistro §7.1) y se
     aplica sin cambios al test.
4. Escribe `results/<exp>/<fecha>_<hash>/raw/seed_<n>.csv` (una fila por composición de test),
   `run_metadata.json`, `metrics_by_seed.csv` y `summary.csv`; copia el resumen a
   `results/<exp>/summary.csv` y añade una fila a `results/runs_index.csv`.

Semillas: de cada semilla salen `fold_assignment` (folds internos) y `model_init` (modelos),
derivadas de forma determinista por esquema y fold, así que el resultado no depende del orden
de ejecución ni del número de procesos.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from perovskite_pu.evaluation.metrics import (
    expected_hits_at_ppr,
    lower_bound_metrics,
    threshold_for_recall,
)
from perovskite_pu.splits.schemes import SCHEMES, Fold
from perovskite_pu.training.baselines import (
    SUBSETS,
    make_estimator,
    model_matrix,
    rule_scores,
)
from perovskite_pu.utils.seeds import sklearn_random_state, spawn_streams

RAW_COLUMNS = ["baseline", "scheme", "seed", "fold", "composition", "label", "score",
               "threshold", "pred"]  # fmt: skip


# ---------------------------------------------------------------------------
# Folds y semillas
# ---------------------------------------------------------------------------
def frozen_folds(
    folds_table: pd.DataFrame, compositions: Sequence[str], scheme: str, seed: int
) -> list[Fold]:
    """Folds de `data/splits/folds.parquet` como índices sobre `compositions`.

    El test de cada fold son sus composiciones; el train, todas las demás de `compositions`
    (en todos los esquemas el train es el complemento del test, también en `temporal`).
    """
    pos = {c: i for i, c in enumerate(compositions)}
    sub = folds_table[(folds_table["scheme"] == scheme) & (folds_table["seed"] == seed)]
    if sub.empty:
        raise KeyError(f"No hay folds congelados para {scheme}, semilla {seed}.")
    unknown = set(sub["composition"]) - set(pos)
    if unknown:
        raise KeyError(f"Composiciones de los folds fuera del pool: {sorted(unknown)[:5]}")
    n, out = len(compositions), []
    for k, g in sub.groupby("fold", sort=True):
        test = np.sort(np.array([pos[c] for c in g["composition"]], dtype=int))
        train = np.setdiff1d(np.arange(n), test)
        groups = tuple(sorted({str(x) for x in g["group"].dropna()}))
        out.append(Fold(train, test, groups or (str(k),)))
    return out


def task_seed(seed: int, stream: str, scheme: str, fold: int) -> np.random.SeedSequence:
    """SeedSequence determinista para (semilla, flujo, esquema, fold).

    Se deriva añadiendo (índice del esquema, fold) a la `spawn_key` del flujo, así que no
    depende del orden de ejecución.
    """
    parent = spawn_streams(seed)[stream]
    key = (*parent.spawn_key, SCHEMES.index(scheme), int(fold))
    return np.random.SeedSequence(entropy=parent.entropy, spawn_key=key)


# ---------------------------------------------------------------------------
# Puntuación de un modelo con umbral calibrado en el bucle interno
# ---------------------------------------------------------------------------
def _score(model: Any, X: np.ndarray) -> np.ndarray:
    """Puntuación continua del estimador (probabilidad de la clase 1 o función de decisión)."""
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1]
    return np.asarray(model.decision_function(X))


def inner_threshold(
    estimator: Any,
    X: np.ndarray,
    s: np.ndarray,
    n_splits: int,
    target_recall: float,
    random_state: int,
) -> float:
    """Umbral a `target_recall` sobre puntuaciones FUERA DE FOLD de un CV interno estratificado.

    Solo usa el train externo (X, s). Cada fold interno ajusta un clon nuevo del estimador.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.empty(len(s), dtype=float)
    for tr, te in skf.split(X, s):
        model = clone(estimator).fit(X[tr], s[tr])
        oof[te] = _score(model, X[te])
    return threshold_for_recall(s, oof, target_recall)


def evaluate_fold(
    inputs: pd.DataFrame,
    labels: np.ndarray,
    fold: Fold,
    fold_index: int,
    scheme: str,
    seed: int,
    baselines: Mapping[str, Mapping[str, Any]],
    inner_cfg: Mapping[str, Any],
) -> pd.DataFrame:
    """Predicciones de todas las líneas base en el test de un fold."""
    rows = []
    comps = inputs.index.to_numpy()
    for name, spec in baselines.items():
        subset = spec["subset"]
        if subset not in SUBSETS:
            raise ValueError(f"{name}: subconjunto desconocido {subset}")
        mask = inputs[subset].to_numpy(dtype=bool)
        tr, te = fold.train_idx[mask[fold.train_idx]], fold.test_idx[mask[fold.test_idx]]
        if te.size == 0:
            continue  # el fold no tiene composiciones de este subconjunto
        if spec["kind"] == "rule":
            scores, thr = rule_scores(inputs[spec["input"]].to_numpy()[te], spec["rule"])
        elif spec["kind"] == "model":
            X = model_matrix(inputs, spec["inputs"])
            rs = sklearn_random_state(task_seed(seed, "model_init", scheme, fold_index))
            est = make_estimator(spec["estimator"], spec["inputs"], spec.get("params", {}), rs)
            inner_rs = sklearn_random_state(task_seed(seed, "fold_assignment", scheme, fold_index))
            thr = inner_threshold(est, X[tr], labels[tr], inner_cfg["n_splits"],
                                  inner_cfg["target_recall"], inner_rs)  # fmt: skip
            model = clone(est).fit(X[tr], labels[tr])  # reentrenado con todo el train externo
            scores = _score(model, X[te])
        else:
            raise ValueError(f"{name}: tipo desconocido {spec['kind']}")
        rows.append(pd.DataFrame({
            "baseline": name, "scheme": scheme, "seed": seed, "fold": fold_index,
            "composition": comps[te], "label": labels[te], "score": scores,
            "threshold": thr, "pred": (scores >= thr).astype(int),
        }))  # fmt: skip
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=RAW_COLUMNS)


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def seed_metrics(raw: pd.DataFrame, fixed_ppr: Sequence[float]) -> pd.DataFrame:
    """Métricas por línea base × esquema × semilla, micro-agregadas sobre folds.

    - `tpr`: Σ aciertos / Σ positivos al umbral de la línea base (punto de operación).
    - `ppr`, `ppr_u`: P(ŷ=1) en todo el test y entre las no etiquetadas (control).
    - `tpr_at_ppr_<x>`: por fold, a P(ŷ=1) exacta con la distribución de puntuaciones del test
      (sin etiquetas) y los empates repartidos; aciertos sumados sobre folds (§7.2–7.3).
    - `lee_liu`: recall² / P(ŷ=1), solo para ordenar (SCAR no se cumple).
    - `*_lower_bound`: con U como negativo; cotas inferiores.
    """
    out = []
    for (b, sch, sd), g in raw.groupby(["baseline", "scheme", "seed"], sort=True):
        s, pred = g["label"].to_numpy(int), g["pred"].to_numpy(int)
        n_pos, n_u = int(s.sum()), int((s == 0).sum())
        tpr = float(pred[s == 1].mean()) if n_pos else np.nan
        ppr = float(pred.mean())
        row = {
            "baseline": b, "scheme": sch, "seed": sd, "n_folds": g["fold"].nunique(),
            "n_test": len(g), "n_pos": n_pos, "n_u": n_u, "tpr": tpr, "ppr": ppr,
            "ppr_u": float(pred[s == 0].mean()) if n_u else np.nan,
            "lee_liu": tpr**2 / ppr if ppr and n_pos else np.nan,
        }  # fmt: skip
        for x in fixed_ppr:
            # Por fold, a P(ŷ=1) exacta con los empates repartidos (valor esperado).
            hits = sum(
                expected_hits_at_ppr(gf["label"].to_numpy(int), gf["score"].to_numpy(float), x)
                for _, gf in g.groupby("fold")
            )
            row[f"tpr_at_ppr_{x:.2f}"] = hits / n_pos if n_pos else np.nan
        if n_pos and n_u:
            row.update(lower_bound_metrics(s, pred, g["score"].to_numpy(float)))
        out.append(row)
    return pd.DataFrame(out)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    """Media, DE e IC 95 % (t de Student sobre semillas) por línea base × esquema.

    Descriptivo: las comparaciones formales (Nadeau–Bengio, bootstrap por grupo) son de F4.
    """
    keys = {"baseline", "scheme", "seed", "n_folds", "n_test", "n_pos", "n_u"}
    value_cols = [c for c in metrics.columns if c not in keys]
    rows = []
    for (b, sch), g in metrics.groupby(["baseline", "scheme"], sort=True):
        n = len(g)
        row = {"baseline": b, "scheme": sch, "n_seeds": n, "n_pos": int(g["n_pos"].iloc[0]),
               "n_u": int(g["n_u"].iloc[0])}  # fmt: skip
        for c in value_cols:
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
def _run_seed_scheme(inputs, labels, folds, scheme, seed, baselines, inner_cfg) -> pd.DataFrame:
    """Todas las líneas base en todos los folds de un esquema y una semilla."""
    parts = [evaluate_fold(inputs, labels, f, k, scheme, seed, baselines, inner_cfg)
             for k, f in enumerate(folds)]  # fmt: skip
    return pd.concat(parts, ignore_index=True)


def run_experiment(config_path: Path | str, allow_dirty: bool = False) -> Path:
    """Ejecuta un experimento de líneas base descrito en un YAML. Devuelve el directorio."""
    import json

    import yaml
    from joblib import Parallel, delayed

    from perovskite_pu.training.baselines import baseline_inputs
    from perovskite_pu.utils.hashing import config_hash
    from perovskite_pu.utils.paths import RESULTS_DIR, ROOT
    from perovskite_pu.utils.run_metadata import (
        build_run_metadata,
        make_run_dir,
        write_run_metadata,
    )
    from perovskite_pu.utils.seeds import load_seed_config

    t0 = time.perf_counter()
    cfg = yaml.safe_load(Path(config_path).read_text("utf-8"))
    seed_cfg = load_seed_config()
    seeds = seed_cfg.active_seeds(cfg["execution"].get("dev_mode", False))
    n_jobs = int(cfg["execution"].get("n_jobs", 1))
    manifest = json.loads((ROOT / "data" / "MANIFEST.json").read_text("utf-8"))
    meta = build_run_metadata(cfg, seeds, n_jobs=n_jobs, allow_dirty=allow_dirty,
                              mp_database_version=manifest.get("mp_database_version"))  # fmt: skip

    pool = pd.read_parquet(ROOT / cfg["data"]["input"])
    folds_table = pd.read_parquet(ROOT / cfg["data"]["folds"])
    inputs = baseline_inputs(pool["composition"])
    labels = pool["label"].to_numpy(int)
    t_load = time.perf_counter()

    tasks = [(scheme, seed, frozen_folds(folds_table, list(pool["composition"]), scheme, seed))
             for seed in seeds for scheme in cfg["splits"]["schemes"]]  # fmt: skip
    parts = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_run_seed_scheme)(inputs, labels, folds, scheme, seed, cfg["baselines"],
                                  cfg["inner_cv"])
        for scheme, seed, folds in tasks
    )  # fmt: skip
    raw = pd.concat(parts, ignore_index=True)
    t_run = time.perf_counter()

    run_dir = make_run_dir(cfg["name"], config_hash(cfg))
    for seed, g in raw.groupby("seed"):
        g.to_csv(run_dir / "raw" / f"seed_{seed}.csv", index=False, lineterminator="\n")
    aux = inputs[["in_bartel2019", *SUBSETS]].reset_index()
    aux.to_csv(run_dir / "raw" / "compositions.csv", index=False, lineterminator="\n")
    metrics = seed_metrics(raw, cfg["evaluation"]["fixed_ppr"])
    summary = summarize(metrics)
    metrics.to_csv(run_dir / "metrics_by_seed.csv", index=False, lineterminator="\n")
    summary.to_csv(run_dir / "summary.csv", index=False, lineterminator="\n")
    summary.to_csv(RESULTS_DIR / cfg["name"] / "summary.csv", index=False, lineterminator="\n")

    meta["timings"] = {"load_s": round(t_load - t0, 3), "run_s": round(t_run - t_load, 3),
                       "total_s": round(time.perf_counter() - t0, 3)}  # fmt: skip
    meta["n_tasks"] = len(tasks)
    write_run_metadata(meta, run_dir)
    _append_index(RESULTS_DIR / "runs_index.csv", cfg["name"], run_dir, meta)
    return run_dir


def _append_index(index_path: Path, experiment: str, run_dir: Path, meta: Mapping) -> None:
    """Añade la corrida a `results/runs_index.csv` (lo crea con cabecera si no existe)."""
    from perovskite_pu.utils.paths import ROOT

    row = pd.DataFrame([{
        "timestamp_utc": meta["timestamp_utc"], "experiment": experiment,
        "run_dir": run_dir.relative_to(ROOT).as_posix(), "git_commit": meta["git_commit"],
        "git_dirty": meta["git_dirty"], "config_hash": meta["config_hash"],
        "n_seeds": len(meta["seeds"]), "n_jobs": meta["n_jobs"],
        "total_s": meta["timings"]["total_s"],
    }])  # fmt: skip
    header = not index_path.exists()
    row.to_csv(index_path, mode="a", header=header, index=False, lineterminator="\n")
