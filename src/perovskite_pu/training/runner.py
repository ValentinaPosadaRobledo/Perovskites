# ATENCIÓN: este módulo solo lee train_pool y los folds congelados; nunca el conjunto de
# prueba reservado (lo verifica tests/test_leakage.py, que revisa también las cadenas).
"""Ejecutor de experimentos: YAML → folds congelados → predicciones por composición → resumen.

Flujo de `run_experiment` (regla 2.2 de CLAUDE.md):
1. Lee el YAML de `configs/experiments/` y las semillas de `configs/seeds.yaml`.
2. Se niega a correr con el árbol de git sucio, salvo `allow_dirty` (guarda el diff).
3. Para cada semilla × esquema × fold (en paralelo con joblib/loky) evalúa cada modelo:
   - reglas: puntuación y umbral externos, sin entrenamiento;
   - modelos simples (`model`): se ajustan SOLO con el train del fold; el umbral es el que da
     recall 0,90 en las puntuaciones fuera de fold de un bucle interno estratificado
     (preregistro §7.1) y se aplica sin cambios al test;
   - bagging PU (`pu_bagging`): CV anidada. En el bucle interno, cada configuración candidata
     (búsqueda aleatoria de presupuesto fijo) se evalúa con los mismos folds internos; se elige
     la de menor P(ŷ=1) fuera de fold a su umbral de recall 0,90, se reentrena con todo el train
     externo y su umbral se aplica sin cambios al test.
4. Cada fold terminado se guarda en `raw/parts/` (la corrida se puede reanudar). Al final
   escribe `raw/seed_<n>.csv` (una fila por composición de test), `raw/search_seed_<n>.csv`
   (resultado interno de cada candidato), `run_metadata.json`, `metrics_by_seed.csv` y
   `summary.csv`; copia el resumen a `results/<exp>/summary.csv` y añade una fila a
   `results/runs_index.csv`.

Semillas: de cada semilla salen `fold_assignment` (folds internos), `pu_sampling` (muestras de
U del bagging) y `model_init` (modelos y sorteo de candidatos), derivadas de forma determinista
por esquema, fold y candidato: el resultado no depende del orden ni del número de procesos.
La columna `baseline` de las salidas identifica el modelo (línea base o modelo PU).
"""

from __future__ import annotations

import itertools
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
from perovskite_pu.training.pu_bagging import make_pu_pipeline
from perovskite_pu.utils.seeds import sklearn_random_state, spawn_streams

RAW_COLUMNS = ["baseline", "scheme", "seed", "fold", "composition", "label", "score",
               "threshold", "pred", "selected"]  # fmt: skip
SEARCH_COLUMNS = ["baseline", "scheme", "seed", "fold", "candidate", "learner", "inner_ppr",
                  "inner_threshold", "selected"]  # fmt: skip
# Índice reservado en la spawn_key para el sorteo de candidatos (no coincide con ningún esquema).
SEARCH_KEY = len(SCHEMES)
# "Esquemas" que no son particiones de train_pool, con su propio índice en la spawn_key (F3.4:
# el modelo final, entrenado con todo train_pool). No cambian las semillas de los esquemas.
EXTRA_SCHEME_KEYS = {"final": SEARCH_KEY + 1}


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


def task_seed(
    seed: int, stream: str, scheme: str, fold: int, *extra: int
) -> np.random.SeedSequence:
    """SeedSequence determinista para (semilla, flujo, esquema, fold[, extra...]).

    Se deriva añadiendo (índice del esquema, fold, extra...) a la `spawn_key` del flujo, así que
    no depende del orden de ejecución. `extra` distingue, p. ej., los candidatos de la búsqueda.
    """
    parent = spawn_streams(seed)[stream]
    idx = SCHEMES.index(scheme) if scheme in SCHEMES else EXTRA_SCHEME_KEYS[scheme]
    key = (*parent.spawn_key, idx, int(fold), *(int(e) for e in extra))
    return np.random.SeedSequence(entropy=parent.entropy, spawn_key=key)


def sample_candidates(search: Mapping[str, Any], seed: int) -> list[dict[str, Any]]:
    """Configuraciones candidatas de la búsqueda aleatoria de presupuesto fijo.

    Se enumeran todas las combinaciones del espacio de cada aprendiz base (`fixed` + producto
    de `space`) y se sortean `n_iter` distintas, sin reemplazo, con el flujo `model_init` de la
    semilla (clave reservada `SEARCH_KEY`). Son las mismas para todos los esquemas y folds de
    esa semilla.
    """
    grid = []
    for learner in sorted(search["learners"]):
        spec = search["learners"][learner]
        keys = sorted(spec.get("space", {}))
        for values in itertools.product(*(spec["space"][k] for k in keys)):
            params = {**spec.get("fixed", {}), **dict(zip(keys, values, strict=True))}
            grid.append({"learner": learner, "params": params})
    parent = spawn_streams(seed)["model_init"]
    rng = np.random.default_rng(
        np.random.SeedSequence(entropy=parent.entropy, spawn_key=(*parent.spawn_key, SEARCH_KEY))
    )
    n = min(int(search["n_iter"]), len(grid))
    return [grid[i] for i in sorted(rng.choice(len(grid), size=n, replace=False))]


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


def select_and_fit_pu(
    spec: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    X_tr: np.ndarray,
    s_tr: np.ndarray,
    X_te: np.ndarray,
    seed: int,
    scheme: str,
    fold_index: int,
    inner_cfg: Mapping[str, Any],
) -> tuple[np.ndarray, float, int, list[dict[str, Any]]]:
    """CV anidada del bagging PU en un fold externo (preregistro §7.1).

    Devuelve (puntuaciones del test, umbral, candidato elegido, registro de la búsqueda).
    Solo usa el train externo para elegir; el test solo se puntúa con el modelo final.
    """
    inner_rs = sklearn_random_state(task_seed(seed, "fold_assignment", scheme, fold_index))
    skf = StratifiedKFold(n_splits=inner_cfg["n_splits"], shuffle=True, random_state=inner_rs)
    splits = list(skf.split(X_tr, s_tr))  # mismos folds internos para todos los candidatos
    log, best = [], None
    for c, cand in enumerate(candidates):
        rs_sampling = sklearn_random_state(task_seed(seed, "pu_sampling", scheme, fold_index, c))
        rs_model = sklearn_random_state(task_seed(seed, "model_init", scheme, fold_index, c))
        est = make_pu_pipeline(cand["learner"], cand["params"], spec.get("bagging", {}),
                               spec.get("preprocessing", {}), rs_sampling, rs_model)  # fmt: skip
        oof = np.empty(len(s_tr), dtype=float)
        for tr, te in splits:
            oof[te] = _score(clone(est).fit(X_tr[tr], s_tr[tr]), X_tr[te])
        thr = threshold_for_recall(s_tr, oof, inner_cfg["target_recall"])
        ppr = float((oof >= thr).mean())  # P(ŷ=1) fuera de fold: criterio de selección
        log.append({"candidate": c, "learner": cand["learner"], "inner_ppr": ppr,
                    "inner_threshold": thr})  # fmt: skip
        if best is None or ppr < best[1]:  # empate: gana el candidato de menor índice
            best = (c, ppr, thr, est)
    c_best, _, thr_best, est_best = best
    for row in log:
        row["selected"] = row["candidate"] == c_best
    model = clone(est_best).fit(X_tr, s_tr)  # reentrenado con todo el train externo
    return _score(model, X_te), thr_best, c_best, log


def evaluate_fold(
    inputs: pd.DataFrame,
    labels: np.ndarray,
    fold: Fold,
    fold_index: int,
    scheme: str,
    seed: int,
    baselines: Mapping[str, Mapping[str, Any]],
    inner_cfg: Mapping[str, Any],
    features: Mapping[str, tuple[np.ndarray, np.ndarray]] | None = None,
    candidates: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    search_log: list | None = None,
) -> pd.DataFrame:
    """Predicciones de todos los modelos en el test de un fold.

    `features[familia] = (X, elegibles)`: matriz alineada con `inputs` y máscara de las
    composiciones que admite la familia (todas en pre_dft; las de post_dft, solo las elegibles).
    `candidates[modelo]`: configuraciones de la búsqueda de cada modelo PU. Si se pasa
    `search_log`, se le añaden los resultados internos de cada candidato.
    """
    rows = []
    comps = inputs.index.to_numpy()
    for name, spec in baselines.items():
        subset = spec["subset"]
        if spec["kind"] == "pu_bagging":
            if subset != "family":
                raise ValueError(f"{name}: un modelo PU se evalúa en `subset: family`.")
            mask = features[spec["family"]][1]
        elif subset not in SUBSETS:
            raise ValueError(f"{name}: subconjunto desconocido {subset}")
        else:
            mask = inputs[subset].to_numpy(dtype=bool)
        tr, te = fold.train_idx[mask[fold.train_idx]], fold.test_idx[mask[fold.test_idx]]
        if te.size == 0:
            continue  # el fold no tiene composiciones de este subconjunto
        selected = -1
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
        elif spec["kind"] == "pu_bagging":
            X = features[spec["family"]][0]
            scores, thr, selected, log = select_and_fit_pu(
                spec, candidates[name], X[tr], labels[tr], X[te], seed, scheme, fold_index,
                inner_cfg,
            )  # fmt: skip
            if search_log is not None:
                search_log += [{"baseline": name, "scheme": scheme, "seed": seed,
                                "fold": fold_index, **r} for r in log]  # fmt: skip
        else:
            raise ValueError(f"{name}: tipo desconocido {spec['kind']}")
        rows.append(pd.DataFrame({
            "baseline": name, "scheme": scheme, "seed": seed, "fold": fold_index,
            "composition": comps[te], "label": labels[te], "score": scores,
            "threshold": thr, "pred": (scores >= thr).astype(int), "selected": selected,
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
    """Todos los modelos en todos los folds de un esquema y una semilla (sin modelos PU)."""
    parts = [evaluate_fold(inputs, labels, f, k, scheme, seed, baselines, inner_cfg)
             for k, f in enumerate(folds)]  # fmt: skip
    return pd.concat(parts, ignore_index=True)


def _part_name(seed: int, scheme: str, fold: int) -> str:
    """Nombre del archivo parcial de un fold externo."""
    return f"{seed}_{scheme}_{fold:03d}"


def _run_task(inputs, labels, fold, k, scheme, seed, models, inner_cfg, features, candidates,
              parts_dir) -> tuple[pd.DataFrame, pd.DataFrame]:  # fmt: skip
    """Un fold externo: evalúa todos los modelos y guarda el resultado parcial en disco."""
    log: list = []
    raw = evaluate_fold(inputs, labels, fold, k, scheme, seed, models, inner_cfg,
                        features, candidates, log)  # fmt: skip
    search = pd.DataFrame(log, columns=SEARCH_COLUMNS)
    if parts_dir is not None:
        name = _part_name(seed, scheme, k)
        search.to_parquet(parts_dir / f"{name}.search.parquet", index=False)
        raw.to_parquet(parts_dir / f"{name}.parquet", index=False)  # el último: marca de fin
    return raw, search


def _family_matrices(models: Mapping[str, Any], compositions: Sequence[str]) -> dict:
    """(X, elegibles) de cada familia de features usada por los modelos PU.

    Las filas no elegibles quedan en NaN y nunca se usan: el modelo se evalúa solo en las
    elegibles (`subset: family`).
    """
    from perovskite_pu.features import sets

    out = {}
    for spec in models.values():
        fam = spec.get("family")
        if spec["kind"] != "pu_bagging" or fam in out:
            continue
        ok = sets.eligible(fam, compositions).to_numpy(bool)
        X_ok = sets.load_feature_matrix(
            fam, [c for c, e in zip(compositions, ok, strict=True) if e]
        )
        X = np.full((len(compositions), X_ok.shape[1]), np.nan)
        X[ok] = X_ok.to_numpy(float)
        out[fam] = (X, ok)
    return out


# Rutas cuyo cambio invalida reanudar una corrida (código, configuración, datos versionados).
RESULT_RELEVANT_PATHS = ("src", "configs", "scripts", "data/splits", "pyproject.toml", "uv.lock")


def code_changed_since(commit: str, repo: Path | None = None) -> list[str]:
    """Archivos que influyen en los resultados y cambiaron entre `commit` y HEAD."""
    import subprocess

    from perovskite_pu.utils.paths import ROOT

    out = subprocess.run(
        ["git", "diff", "--name-only", commit, "HEAD", "--", *RESULT_RELEVANT_PATHS],
        cwd=repo or ROOT, capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    return [ln for ln in out.splitlines() if ln.strip()]


def run_experiment(
    config_path: Path | str,
    allow_dirty: bool = False,
    resume: Path | str | None = None,
    dev_mode: bool | None = None,
) -> Path:
    """Ejecuta un experimento descrito en un YAML y devuelve el directorio de la corrida.

    - `dev_mode` (si no es None) sustituye a `execution.dev_mode` del YAML: permite lanzar la
      primera corrida con 3 semillas sin editar la configuración versionada.
    - Con `resume` (directorio de una corrida interrumpida o de desarrollo) solo ejecuta los
      folds que faltan. Exige la misma configuración (sin `execution`) y que, desde el commit de
      la corrida, no haya cambiado nada que influya en los resultados (`RESULT_RELEVANT_PATHS`).
    """
    import json

    import yaml
    from joblib import Parallel, delayed

    from perovskite_pu.training.baselines import baseline_inputs
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
    models = cfg.get("models", cfg.get("baselines"))
    seed_cfg = load_seed_config()
    dev = cfg["execution"].get("dev_mode", False) if dev_mode is None else dev_mode
    seeds = seed_cfg.active_seeds(dev)
    n_jobs = int(cfg["execution"].get("n_jobs", 1))
    manifest = json.loads((ROOT / "data" / "MANIFEST.json").read_text("utf-8"))
    meta = build_run_metadata(cfg, seeds, n_jobs=n_jobs, allow_dirty=allow_dirty,
                              mp_database_version=manifest.get("mp_database_version"))  # fmt: skip
    # El hash identifica lo que determina los resultados: se excluye `execution` (núcleos,
    # modo de desarrollo), que no los cambia. Así, una corrida con 3 semillas (dev_mode) se
    # puede ampliar a las 10 reanudándola, sin recalcular los folds ya hechos.
    meta["config_hash"] = config_hash({k: v for k, v in cfg.items() if k != "execution"})

    if resume is None:
        run_dir = make_run_dir(cfg["name"], meta["config_hash"])
    else:
        run_dir = Path(resume).resolve()  # admite rutas relativas al directorio actual
        prev = json.loads((run_dir / "run_metadata.json").read_text("utf-8"))
        if prev["config_hash"] != meta["config_hash"]:
            raise RuntimeError("No se puede reanudar: cambió la configuración.")
        changed = code_changed_since(prev["git_commit"])
        if changed:
            raise RuntimeError(
                f"No se puede reanudar: cambiaron archivos relevantes: {changed[:5]}"
            )
        meta["timestamp_utc"] = prev["timestamp_utc"]
        meta["resumed_from_commit"] = prev["git_commit"]
        meta["seeds_previous_runs"] = prev.get("seeds")
    parts_dir = run_dir / "raw" / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    meta["dev_mode"] = bool(dev)
    meta["status"] = "running"
    write_run_metadata(dict(meta), run_dir)  # copia: el diff se conserva para el final

    pool = pd.read_parquet(ROOT / cfg["data"]["input"])
    comps = list(pool["composition"])
    folds_table = pd.read_parquet(ROOT / cfg["data"]["folds"])
    inputs = baseline_inputs(comps)
    labels = pool["label"].to_numpy(int)
    features = _family_matrices(models, comps)
    search = {n: s["search"] for n, s in models.items() if s["kind"] == "pu_bagging"}
    candidates_by_seed = {sd: {n: sample_candidates(sc, sd) for n, sc in search.items()}
                          for sd in seeds}  # fmt: skip
    meta["candidates"] = {str(sd): c for sd, c in candidates_by_seed.items()}
    t_load = time.perf_counter()

    tasks = [(scheme, seed, k, f) for seed in seeds for scheme in cfg["splits"]["schemes"]
             for k, f in enumerate(frozen_folds(folds_table, comps, scheme, seed))]  # fmt: skip
    pending = [
        t for t in tasks if not (parts_dir / f"{_part_name(t[1], t[0], t[2])}.parquet").exists()
    ]
    print(f"{len(tasks)} folds externos; pendientes: {len(pending)}", flush=True)
    results = Parallel(n_jobs=n_jobs, backend="loky", return_as="generator_unordered")(
        delayed(_run_task)(inputs, labels, f, k, scheme, seed, models, cfg["inner_cv"], features,
                           candidates_by_seed[seed], parts_dir)
        for scheme, seed, k, f in pending
    )  # fmt: skip
    for done, _ in enumerate(results, start=1):
        if done % max(1, len(pending) // 50) == 0 or done == len(pending):
            el = time.perf_counter() - t_load
            eta = el / done * (len(pending) - done)
            print(f"  {done}/{len(pending)} folds · {el / 60:.1f} min · faltan ~{eta / 60:.1f} min",
                  flush=True)  # fmt: skip
    t_run = time.perf_counter()

    # Reúne las partes (incluidas las de una ejecución anterior si se reanudó).
    names = [_part_name(seed, scheme, k) for scheme, seed, k, _ in tasks]
    raw = pd.concat([pd.read_parquet(parts_dir / f"{n}.parquet") for n in names], ignore_index=True)
    srch = [pd.read_parquet(parts_dir / f"{n}.search.parquet") for n in names]
    srch = pd.concat([x for x in srch if len(x)] or [pd.DataFrame(columns=SEARCH_COLUMNS)],
                     ignore_index=True)  # fmt: skip
    for seed, g in raw.groupby("seed"):
        g.to_csv(run_dir / "raw" / f"seed_{seed}.csv", index=False, lineterminator="\n")
    for seed, g in srch.groupby("seed"):
        g.to_csv(run_dir / "raw" / f"search_seed_{seed}.csv", index=False, lineterminator="\n")
    aux = inputs[["in_bartel2019", *SUBSETS]].reset_index()
    for fam, (_, ok) in features.items():
        aux[f"eligible_{fam}"] = ok
    aux.to_csv(run_dir / "raw" / "compositions.csv", index=False, lineterminator="\n")
    metrics = seed_metrics(raw, cfg["evaluation"]["fixed_ppr"])
    summary = summarize(metrics)
    metrics.to_csv(run_dir / "metrics_by_seed.csv", index=False, lineterminator="\n")
    summary.to_csv(run_dir / "summary.csv", index=False, lineterminator="\n")
    summary.to_csv(RESULTS_DIR / cfg["name"] / "summary.csv", index=False, lineterminator="\n")
    if len(srch):
        sel = srch[srch["selected"]].groupby(["baseline", "scheme", "learner"]).size()
        sel.rename("n_folds").reset_index().to_csv(
            run_dir / "selected_learners.csv", index=False, lineterminator="\n"
        )

    meta["status"] = "completed"
    meta["timings"] = {"load_s": round(t_load - t0, 3), "run_s": round(t_run - t_load, 3),
                       "total_s": round(time.perf_counter() - t0, 3),
                       "resumed": resume is not None}  # fmt: skip
    meta["n_tasks"] = len(tasks)
    write_run_metadata(meta, run_dir)
    append_runs_index(RESULTS_DIR / "runs_index.csv", cfg["name"], run_dir, meta)
    return run_dir
