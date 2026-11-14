"""F3.2: ejecución de la estimación de la prior (YAML → semillas × métodos × poblaciones).

Lee `configs/experiments/f3_2_prior.yaml` y estima c, α y π_U en `train_pool` con cada
método y población, para las 10 semillas. Escribe, como el resto de experimentos (regla 2.2):

    results/f3_2_prior/<timestamp>_<hash>/
        raw/seed_<n>.csv            una fila por método × población
        raw/en_scores_seed_<n>.csv  g(x) fuera de fold del Elkan–Noto principal (diagnóstico)
        run_metadata.json, summary.csv
    results/f3_2_prior/summary.csv y una fila en results/runs_index.csv

Semillas: de cada semilla, `fold_assignment` fija los folds (validación cruzada de
Elkan–Noto y folds árbol/estimación de TIcE) y `model_init` el clasificador de Elkan–Noto.
`pu_sampling` no se usa (no hay muestreo de U).

No hay test externo: π es una propiedad de `train_pool` y se estima con todas sus filas. Aun
así, Elkan–Noto puntúa cada positiva con un modelo que no la vio (validación cruzada), y su
imputación y calibración se ajustan dentro de cada fold (`Pipeline`, regla 2.1.6).
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from perovskite_pu.evaluation.prior import (
    elkan_noto_label_frequency,
    prior_from_label_frequency,
    tice_label_frequency,
)
from perovskite_pu.training.pu_bagging import make_base_estimator
from perovskite_pu.utils.seeds import sklearn_random_state, spawn_streams

VALUE_COLUMNS = ("c", "alpha", "pi_u", "hidden_positives")


# ---------------------------------------------------------------------------
# Piezas
# ---------------------------------------------------------------------------
def _imputer(preprocessing: Mapping[str, Any]) -> SimpleImputer:
    """Imputación por la mediana (con indicador de hueco si se pide), como en F3.3."""
    return SimpleImputer(strategy="median",
                         add_indicator=bool(preprocessing.get("add_missing_indicator")),
                         keep_empty_features=False)  # fmt: skip


def make_elkan_noto_estimator(
    spec: Mapping[str, Any], preprocessing: Mapping[str, Any], random_state: int
) -> Pipeline:
    """Imputación → umbral de varianza → clasificador base calibrado (P(s = 1 | x)).

    La calibración (`CalibratedClassifierCV`) es imprescindible: Elkan–Noto usa el VALOR de
    la probabilidad. Todo se ajusta solo con el train de cada fold de Elkan–Noto.
    """
    base = make_base_estimator(spec["learner"], spec.get("params", {}))
    base.set_params(random_state=random_state)
    cal = spec.get("calibration", {})
    model = CalibratedClassifierCV(base, method=cal.get("method", "isotonic"),
                                   cv=int(cal.get("cv", 3)))  # fmt: skip
    return Pipeline([("impute", _imputer(preprocessing)),
                     ("variance", VarianceThreshold(threshold=0.0)),
                     ("model", model)])  # fmt: skip


def population_mask(pool: pd.DataFrame, spec: Mapping[str, Any]) -> np.ndarray:
    """Filas de la población: todas las P y las U que no tengan la bandera excluida."""
    flag = spec.get("exclude_unlabeled_flag")
    keep = np.ones(len(pool), bool)
    if flag:
        keep &= ~((pool["label"].to_numpy(int) == 0) & pool[flag].to_numpy(bool))
    return keep


def estimate_one(
    X: np.ndarray,
    s: np.ndarray,
    method: str,
    spec: Mapping[str, Any],
    preprocessing: Mapping[str, Any],
    seed: int,
) -> tuple[dict[str, Any], np.ndarray | None]:
    """Una estimación (método × semilla) en una población ya filtrada.

    Devuelve la fila de resultados y, para Elkan–Noto, las puntuaciones g(x) fuera de fold.
    """
    streams = spawn_streams(seed)
    fold_rs = sklearn_random_state(streams["fold_assignment"])
    model_rs = sklearn_random_state(streams["model_init"])
    extra: dict[str, Any] = {}
    oof = None
    if spec["kind"] == "elkan_noto":
        est = make_elkan_noto_estimator(spec, preprocessing, model_rs)
        c, oof = elkan_noto_label_frequency(X, s, est, n_splits=int(spec["n_splits"]),
                                            random_state=fold_rs)  # fmt: skip
        # Diagnóstico de calibración: la media de g en todas las filas debe ≈ n_L / N.
        extra = {"g_mean_all": float(np.mean(oof)), "g_mean_unlabeled": float(oof[s == 0].mean())}
    elif spec["kind"] == "tice":
        # TIcE no admite huecos: se imputa con toda la población (no hay test que proteger;
        # los folds árbol/estimación los hace el propio método).
        Xi = _imputer(preprocessing).fit_transform(X)
        c, detail = tice_label_frequency(
            Xi, s, n_folds=int(spec["n_folds"]), k=float(spec["k"]),
            max_splits=int(spec["max_splits"]), n_bins=int(spec["n_bins"]),
            binning=spec["binning"], random_state=fold_rs,
        )  # fmt: skip
        extra = {"c_first_pass": float(np.mean(detail["first_pass"])),
                 "c_fold_sd": float(np.std(detail["second_pass"], ddof=1))}  # fmt: skip
    else:
        raise ValueError(f"Método de prior desconocido: {spec['kind']}")
    prior = prior_from_label_frequency(c, int(s.sum()), int((s == 0).sum()))
    row = {"method": method, "kind": spec["kind"], "role": spec.get("role"), "seed": seed,
           "c": prior.c, "alpha": prior.alpha, "pi_u": prior.pi_u,
           "hidden_positives": prior.hidden_positives, "clipped": prior.clipped,
           "n_labeled": prior.n_labeled, "n_unlabeled": prior.n_unlabeled, **extra}  # fmt: skip
    return row, oof


def summarize_prior(raw: pd.DataFrame) -> pd.DataFrame:
    """Media, DE, IC 95 % (t de Student sobre semillas), mín. y máx. por método × población."""
    rows = []
    for (pop, method), g in raw.groupby(["population", "method"], sort=True):
        n = len(g)
        row = {"population": pop, "method": method, "kind": g["kind"].iloc[0],
               "role": g["role"].iloc[0], "n_seeds": n, "n_labeled": int(g["n_labeled"].iloc[0]),
               "n_unlabeled": int(g["n_unlabeled"].iloc[0]),
               "n_clipped": int(g["clipped"].sum())}  # fmt: skip
        for col in VALUE_COLUMNS:
            v = g[col].astype(float)
            m, sd = float(v.mean()), float(v.std(ddof=1)) if n > 1 else 0.0
            half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else 0.0
            row.update({f"{col}_mean": m, f"{col}_sd": sd, f"{col}_ci95_low": m - half,
                        f"{col}_ci95_high": m + half, f"{col}_min": float(v.min()),
                        f"{col}_max": float(v.max())})  # fmt: skip
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Ejecución completa
# ---------------------------------------------------------------------------
def _task(X, s, comps, pop, method, spec, preprocessing, seed, keep_scores):
    """Tarea paralela: una estimación; devuelve la fila y, si se piden, las puntuaciones."""
    row, oof = estimate_one(X, s, method, spec, preprocessing, seed)
    row["population"] = pop
    scores = None
    if keep_scores and oof is not None:
        scores = pd.DataFrame({"seed": seed, "composition": comps, "label": s, "g": oof})
    return row, scores


def run_prior_experiment(
    config_path: Path | str, allow_dirty: bool = False, dev_mode: bool | None = None
) -> Path:
    """Ejecuta F3.2 desde su YAML y devuelve el directorio de la corrida."""
    import json

    import yaml
    from joblib import Parallel, delayed

    from perovskite_pu.features import sets
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
    dev = cfg["execution"].get("dev_mode", False) if dev_mode is None else dev_mode
    seeds = load_seed_config().active_seeds(dev)
    n_jobs = int(cfg["execution"].get("n_jobs", 1))
    manifest = json.loads((ROOT / "data" / "MANIFEST.json").read_text("utf-8"))
    meta = build_run_metadata(cfg, seeds, n_jobs=n_jobs, allow_dirty=allow_dirty,
                              mp_database_version=manifest.get("mp_database_version"))  # fmt: skip
    # Como en el ejecutor de F3.3: `execution` (núcleos, modo de desarrollo) no entra en el hash.
    meta["config_hash"] = config_hash({k: v for k, v in cfg.items() if k != "execution"})
    meta["dev_mode"] = bool(dev)
    meta["status"] = "running"
    run_dir = make_run_dir(cfg["name"], meta["config_hash"])
    write_run_metadata(dict(meta), run_dir)

    pool = pd.read_parquet(ROOT / cfg["data"]["input"])
    comps = pool["composition"].tolist()
    X_all = sets.load_feature_matrix(cfg["features"]["family"], comps).to_numpy(float)
    s_all = pool["label"].to_numpy(int)
    prep = cfg["features"].get("preprocessing", {})
    # Las puntuaciones g(x) se guardan solo del Elkan–Noto principal (diagnóstico).
    principal_en = next(
        m for m, sp in cfg["methods"].items()
        if sp["kind"] == "elkan_noto" and sp.get("role") == "principal"
    )  # fmt: skip
    t_load = time.perf_counter()

    tasks = []
    for pop, pop_spec in cfg["populations"].items():
        keep = population_mask(pool, pop_spec)
        X, s = X_all[keep], s_all[keep]
        cp = [c for c, k in zip(comps, keep, strict=True) if k]
        for method, spec in cfg["methods"].items():
            for seed in seeds:
                keep_scores = method == principal_en and pop == "all"
                tasks.append((X, s, cp, pop, method, spec, prep, seed, keep_scores))
    out = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_task)(*t) for t in tasks)
    t_run = time.perf_counter()

    raw = pd.DataFrame([r for r, _ in out])
    for seed, g in raw.groupby("seed"):
        g.to_csv(run_dir / "raw" / f"seed_{seed}.csv", index=False, lineterminator="\n")
    scores = [sc for _, sc in out if sc is not None]
    for sc in scores:
        seed = int(sc["seed"].iloc[0])
        sc.to_csv(run_dir / "raw" / f"en_scores_seed_{seed}.csv", index=False,
                  lineterminator="\n")  # fmt: skip
    summary = summarize_prior(raw)
    summary.to_csv(run_dir / "summary.csv", index=False, lineterminator="\n")
    summary.to_csv(RESULTS_DIR / cfg["name"] / "summary.csv", index=False, lineterminator="\n")

    meta["status"] = "completed"
    meta["n_tasks"] = len(tasks)
    meta["timings"] = {"load_s": round(t_load - t0, 3), "run_s": round(t_run - t_load, 3),
                       "total_s": round(time.perf_counter() - t0, 3)}  # fmt: skip
    write_run_metadata(meta, run_dir)
    append_runs_index(RESULTS_DIR / "runs_index.csv", cfg["name"], run_dir, meta)
    return run_dir
