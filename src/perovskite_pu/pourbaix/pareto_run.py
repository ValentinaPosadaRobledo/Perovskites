"""F5.2: ejecución del filtro de toxicidad y del frente de Pareto desde su YAML.

Lee la última corrida de F5.1 (`pourbaix.csv`) y los `raw/` de la última corrida de F3.3 (las
10 semillas) y escribe en results/f5_2_pareto/<timestamp>_<hash>/:

- `candidates.csv`: condición principal; una fila por composición con puntuación, ΔG_pbx,
  marcas de toxicidad y rango de Pareto (P y marcadas con rango 0).
- `front.csv`: el frente principal (rango 1), ordenado por puntuación.
- `sensitivity.csv`: rango de cada composición en todas las condiciones de F5.1 y en cuántas
  queda en el frente.
- `summary.csv`: recuentos por condición (también en results/f5_2_pareto/summary.csv).
- `run_metadata.json` y una fila en results/runs_index.csv.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from perovskite_pu.pourbaix.pareto import condition_front, oof_scores


def _last_run(index: pd.DataFrame, experiment: str, root: Path) -> Path:
    """Carpeta de la última corrida de un experimento en `runs_index.csv`."""
    rows = index[index["experiment"] == experiment]
    if rows.empty:
        raise FileNotFoundError(f"No hay corridas de {experiment} en runs_index.csv.")
    return root / rows.iloc[-1]["run_dir"]


def run_pareto_experiment(config_path: Path | str, allow_dirty: bool = False) -> Path:
    """Ejecuta F5.2 desde su YAML y devuelve el directorio de la corrida."""
    import yaml

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
    inp = cfg["inputs"]
    index = pd.read_csv(RESULTS_DIR / "runs_index.csv")
    pbx_dir = _last_run(index, inp["pourbaix_experiment"], ROOT)
    score_dir = _last_run(index, inp["score_experiment"], ROOT)
    seeds = list(load_seed_config().seeds)
    paths = [score_dir / "raw" / f"seed_{s}.csv" for s in seeds]
    digest = hashlib.sha256()
    for p in paths:
        digest.update(p.read_bytes())

    full_cfg = {**cfg, "resolved_inputs": {
        "pourbaix_run": pbx_dir.relative_to(ROOT).as_posix(),
        "score_run": score_dir.relative_to(ROOT).as_posix(),
        "score_raw_sha256": digest.hexdigest()}}  # fmt: skip
    meta = build_run_metadata(full_cfg, [], n_jobs=1, allow_dirty=allow_dirty)
    meta["config_hash"] = config_hash(full_cfg)
    meta["status"] = "running"
    run_dir = make_run_dir(cfg["name"], meta["config_hash"])
    write_run_metadata(dict(meta), run_dir)

    raw = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    scores = oof_scores(raw, inp["score_model"], inp["score_scheme"])
    pbx = pd.read_csv(pbx_dir / "pourbaix.csv")
    table = pbx.merge(scores, on="composition", how="left", validate="many_to_one")
    if table["score_mean"].isna().any():
        missing = table.loc[table["score_mean"].isna(), "composition"].unique()[:5]
        raise ValueError(f"Composiciones de F5.1 sin puntuación de F3.3: {list(missing)}")

    tox = cfg["toxicity"]
    conds = table[["pH", "ion_conc_M"]].drop_duplicates().sort_values(["pH", "ion_conc_M"])
    per_cond, summary = [], []
    for ph, conc in conds.itertuples(index=False):
        sub = condition_front(table, ph, conc, tox["elements"], tox["aqueous_species"])
        per_cond.append(sub)
        summary.append({"pH": ph, "ion_conc_M": conc, "n_compositions": len(sub),
                        "n_unlabeled": int((sub["label"] == 0).sum()),
                        "n_flagged_toxic": int((sub["toxicity"] != "").sum()),
                        "n_unlabeled_eligible": int((sub["pareto_rank"] > 0).sum()),
                        "n_front": int(sub["on_front"].sum()),
                        "n_front_below_ref": int((sub["on_front"] & (sub["dg_max"] <= float(
                            cfg["reference"]["dg_threshold_eV_atom"]))).sum())})  # fmt: skip
    allc = pd.concat(per_cond, ignore_index=True)
    pc = cfg["principal_condition"]
    main = allc[np.isclose(allc["pH"], pc["pH"]) & np.isclose(allc["ion_conc_M"], pc["ion_conc_M"])]
    keep = ["composition", "label", "A", "B", "score_mean", "score_sd", "frac_above_threshold",
            "dg_max", "v_at_max", "dg_min", "stable_at_max", "stable_in_window", "toxicity",
            "pareto_rank", "on_front"]  # fmt: skip
    sens = allc.pivot_table(index="composition", columns=["pH", "ion_conc_M"],
                            values="pareto_rank", aggfunc="first")  # fmt: skip
    sens.columns = [f"rank_pH{p:g}_c{c:.0e}" for p, c in sens.columns]
    sens["n_conditions_on_front"] = (sens == 1).sum(axis=1)

    lt = {"lineterminator": "\n", "index": False}
    main[keep].sort_values("composition").to_csv(run_dir / "candidates.csv", **lt)
    main[main["on_front"]][keep].sort_values("score_mean", ascending=False).to_csv(
        run_dir / "front.csv", **lt
    )
    sens.reset_index().to_csv(run_dir / "sensitivity.csv", **lt)
    summ = pd.DataFrame(summary)
    summ.to_csv(run_dir / "summary.csv", **lt)
    (RESULTS_DIR / cfg["name"]).mkdir(parents=True, exist_ok=True)
    summ.to_csv(RESULTS_DIR / cfg["name"] / "summary.csv", **lt)

    meta["status"] = "completed"
    meta["counts"] = {k: int(v) for k, v in summ.set_index(["pH", "ion_conc_M"]).loc[
        (pc["pH"], pc["ion_conc_M"])].items()}  # fmt: skip
    meta["timings"] = {"total_s": round(time.perf_counter() - t0, 3)}
    write_run_metadata(meta, run_dir)
    append_runs_index(RESULTS_DIR / "runs_index.csv", cfg["name"], run_dir, meta)
    print(json.dumps(meta["counts"], indent=2))
    return run_dir
