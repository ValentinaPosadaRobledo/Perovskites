"""F4.3: ejecución del análisis SHAP desde su YAML (`configs/experiments/f4_3_shap.yaml`).

Por semilla: búsqueda interna y selección como en F3.3 sobre todo `train_pool` (variante
`final` de F3.4), reconstrucción del candidato elegido con sus mismas semillas (se comprueba
que puntúa igual), TreeSHAP *path-dependent* promediado sobre las bolsas y comprobación de
aditividad (ver `shap_analysis`).

Salidas: results/f4_3_shap/<timestamp>_<hash>/ con raw/shap_seed_<n>.parquet (SHAP por
composición; no versionado), importance_by_seed.csv, group_importance_by_seed.csv,
selected.csv, stability_{features,groups}_pairs.csv, summary_{features,groups}.csv,
summary.csv (grupos) y run_metadata.json; copia del resumen en results/f4_3_shap/summary.csv y
fila en results/runs_index.csv.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from perovskite_pu.evaluation.shap_analysis import (
    bagging_shap,
    feature_groups,
    importance_tables,
    stability,
)


def fit_final_model(
    spec: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    X: np.ndarray,
    s: np.ndarray,
    seed: int,
    inner_cfg: Mapping[str, Any],
) -> tuple[Any, int, float]:
    """Modelo final de una semilla: (pipeline ajustado, candidato elegido, umbral).

    Usa `select_and_fit_pu` con el esquema `final` (las mismas semillas que F3.4) y reconstruye
    el candidato elegido con sus semillas de bolsas e inicialización.

    Raises:
        RuntimeError: si el pipeline reconstruido no puntúa igual que el de la selección.
    """
    from perovskite_pu.training.pu_bagging import make_pu_pipeline
    from perovskite_pu.training.runner import select_and_fit_pu, task_seed
    from perovskite_pu.utils.seeds import sklearn_random_state

    scores, thr, c_best, _ = select_and_fit_pu(spec, candidates, X, s, X, seed, "final", 0,
                                               inner_cfg)  # fmt: skip
    cand = candidates[c_best]
    est = make_pu_pipeline(
        cand["learner"], cand["params"], spec.get("bagging", {}), spec.get("preprocessing", {}),
        sklearn_random_state(task_seed(seed, "pu_sampling", "final", 0, c_best)),
        sklearn_random_state(task_seed(seed, "model_init", "final", 0, c_best)),
    )  # fmt: skip
    est.fit(X, s)
    if not np.allclose(np.asarray(est.predict_proba(X))[:, 1], scores, rtol=0, atol=1e-12):
        raise RuntimeError(f"Semilla {seed}: el modelo reconstruido no coincide con el elegido.")
    return est, c_best, float(thr)


def _seed_task(spec, cands, X, s, names, groups_cfg, shap_cfg, seed, inner_cfg, comps):
    """SHAP de la semilla: tablas de importancia, elección y SHAP por composición."""
    t0 = time.perf_counter()
    est, c_best, thr = fit_final_model(spec, cands, X, s, seed, inner_cfg)
    values, base, out, score, space = bagging_shap(est, X, names)
    gap = float(np.max(np.abs(base + values.sum(axis=1).to_numpy() - out)))
    if gap > float(shap_cfg["additivity_tol"]):
        raise RuntimeError(f"Semilla {seed}: SHAP no aditivo (desvío máximo {gap:.2e}).")
    groups = feature_groups(list(values.columns), groups_cfg)
    feat, grp = importance_tables(values, groups, s)
    selected = {"seed": seed, "candidate": c_best, "learner": cands[c_best]["learner"],
                "threshold": thr, "output_space": space, "base_value": base,
                "additivity_max_gap": gap,
                "spearman_output_score": float(stats.spearmanr(out, score).statistic),
                "n_features_model": values.shape[1],
                "seconds": round(time.perf_counter() - t0, 1)}  # fmt: skip
    per_comp = values.assign(composition=comps, label=s, output=out, score=score, seed=seed)
    return feat.assign(seed=seed), grp.assign(seed=seed), selected, per_comp


def run_shap_experiment(
    config_path: Path | str, allow_dirty: bool = False, dev_mode: bool | None = None
) -> Path:
    """Ejecuta F4.3 desde su YAML y devuelve el directorio de la corrida."""
    import json

    import yaml
    from joblib import Parallel, delayed

    from perovskite_pu.training.runner import _family_matrices, sample_candidates
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
    # El modelo se toma de su experimento de origen y se guarda resuelto: el hash cambia si
    # cambia allí, aunque este YAML no cambie.
    src = yaml.safe_load((ROOT / cfg["model_source"]).read_text("utf-8"))
    resolved = {"model": src["models"][cfg["model"]], "inner_cv": src["inner_cv"]}
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

    # Datos: train_pool y su matriz pre_dft (definida en todas las composiciones).
    pool = pd.read_parquet(ROOT / cfg["data"]["input"])
    comps = pool["composition"].tolist()
    s = pool["label"].to_numpy(int)
    spec = resolved["model"]
    from perovskite_pu.features import sets

    names = list(sets.load_feature_matrix(spec["family"], comps[:1]).columns)
    X, ok = _family_matrices({cfg["model"]: spec}, comps)[spec["family"]]
    if not ok.all():
        raise ValueError(f"{cfg['model']}: F4.3 solo admite familias definidas en todo train_pool.")
    t_load = time.perf_counter()

    tasks = [delayed(_seed_task)(spec, sample_candidates(spec["search"], seed), X, s, names,
                                 cfg["groups"], cfg["shap"], seed, resolved["inner_cv"], comps)
             for seed in seeds]  # fmt: skip
    out = Parallel(n_jobs=n_jobs, backend="loky")(tasks)
    t_run = time.perf_counter()

    feat = pd.concat([o[0] for o in out], ignore_index=True)
    grp = pd.concat([o[1] for o in out], ignore_index=True)
    selected = pd.DataFrame([o[2] for o in out])
    for o in out:
        seed = int(o[3]["seed"].iloc[0])
        o[3].to_parquet(run_dir / "raw" / f"shap_seed_{seed}.parquet", index=False)
    st = cfg["stability"]
    pairs_f, sum_f = stability(feat, "feature", st["rank_by"], int(st["top_k_features"]))
    pairs_g, sum_g = stability(grp, "group", st["rank_by"], int(st["top_k_groups"]))
    # Dirección del efecto (media con signo en P y en U) y tamaño de cada grupo, por elemento.
    sign_cols = ["mean_signed_P", "mean_signed_U", "mean_abs_P", "mean_abs_U"]
    sum_f = sum_f.merge(feat.groupby("feature")[sign_cols].mean().reset_index(), on="feature")
    sum_f.insert(1, "group", sum_f["feature"].map(feat.drop_duplicates("feature")
                                                  .set_index("feature")["group"]))  # fmt: skip
    sum_g = sum_g.merge(grp.groupby("group")[[*sign_cols, "n_features"]].mean().reset_index(),
                        on="group")  # fmt: skip

    lt = {"lineterminator": "\n", "index": False}
    feat.to_csv(run_dir / "importance_by_seed.csv", **lt)
    grp.to_csv(run_dir / "group_importance_by_seed.csv", **lt)
    selected.to_csv(run_dir / "selected.csv", **lt)
    pairs_f.to_csv(run_dir / "stability_features_pairs.csv", **lt)
    pairs_g.to_csv(run_dir / "stability_groups_pairs.csv", **lt)
    sum_f.to_csv(run_dir / "summary_features.csv", **lt)
    sum_g.to_csv(run_dir / "summary_groups.csv", **lt)
    sum_g.to_csv(run_dir / "summary.csv", **lt)
    (RESULTS_DIR / cfg["name"]).mkdir(parents=True, exist_ok=True)
    sum_g.to_csv(RESULTS_DIR / cfg["name"] / "summary.csv", **lt)

    meta["status"] = "completed"
    meta["n_tasks"] = len(tasks)
    meta["n_features_model"] = int(selected["n_features_model"].iloc[0])
    meta["timings"] = {"load_s": round(t_load - t0, 3), "run_s": round(t_run - t_load, 3),
                       "total_s": round(time.perf_counter() - t0, 3)}  # fmt: skip
    write_run_metadata(meta, run_dir)
    append_runs_index(RESULTS_DIR / "runs_index.csv", cfg["name"], run_dir, meta)
    return run_dir
