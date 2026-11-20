"""F5.1: ejecución del cálculo de ΔG_pbx desde su YAML (`configs/experiments/f5_1_pourbaix.yaml`).

Lee las entradas descargadas (`data/raw/pourbaix/`, registradas en `data/MANIFEST.json`) y
escribe en results/f5_1_pourbaix/<timestamp>_<hash>/:

- `pourbaix.csv`: una fila por composición × pH × concentración (ΔG_pbx máximo y mínimo en la
  ventana, sus potenciales y la fase estable en el potencial del máximo). Solo composiciones,
  etiquetas y valores derivados.
- `excluded.csv`: composiciones sin entrada perovskita consistente entre las de Pourbaix.
- `summary.csv`: por etiqueta (P, U), pH y concentración: n, mediana y cuartiles de ΔG_pbx
  máximo y fracción por debajo de la referencia (también copiada en results/f5_1_pourbaix/).
- `run_metadata.json`, y una fila en results/runs_index.csv.

Determinista: no usa semillas (se declara con `seeds: []`).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from perovskite_pu.pourbaix.stability import composition_rows, find_target, load_entries


def _composition_task(row: dict[str, Any], entries_dir: Path, cond: dict) -> dict[str, Any]:
    """ΔG_pbx de una composición, o el motivo de su exclusión."""
    entries = load_entries(entries_dir / f"{row['system']}.json.gz")
    target = find_target(entries, row["material_id_gga"])
    if target is None:
        return {"excluded": {**row, "reason": "sin entrada perovskita entre las de Pourbaix"}}
    rows = composition_rows(entries, target, row["A"], row["B"], cond)
    return {"rows": [{**row, "pourbaix_entry_id": str(target.entry_id), **r} for r in rows]}


def summarize(table: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Resumen por etiqueta, pH y concentración del ΔG_pbx máximo (peor caso)."""
    g = table.assign(below=table["dg_max"] <= threshold).groupby(["label", "pH", "ion_conc_M"])
    out = g.agg(n=("dg_max", "size"), dg_max_median=("dg_max", "median"),
                dg_max_q25=("dg_max", lambda s: s.quantile(0.25)),
                dg_max_q75=("dg_max", lambda s: s.quantile(0.75)),
                frac_dg_max_below_ref=("below", "mean"),
                dg_min_median=("dg_min", "median"))  # fmt: skip
    return out.reset_index().assign(reference_eV_atom=threshold)


def run_pourbaix_experiment(config_path: Path | str, allow_dirty: bool = False) -> Path:
    """Ejecuta F5.1 desde su YAML y devuelve el directorio de la corrida."""
    import yaml
    from joblib import Parallel, delayed

    from perovskite_pu.utils.hashing import config_hash, sha256_file
    from perovskite_pu.utils.paths import DATA_DIR, RAW_DIR, RESULTS_DIR, ROOT
    from perovskite_pu.utils.run_metadata import (
        append_runs_index,
        build_run_metadata,
        make_run_dir,
        write_run_metadata,
    )

    t0 = time.perf_counter()
    cfg = yaml.safe_load(Path(config_path).read_text("utf-8"))
    data_cfg = yaml.safe_load((ROOT / cfg["data"]["config"]).read_text("utf-8"))
    manifest = json.loads((DATA_DIR / "MANIFEST.json").read_text("utf-8"))
    pbx = manifest.get("pourbaix")
    if pbx is None:
        raise FileNotFoundError("Sin sección `pourbaix` en data/MANIFEST.json: falta la descarga.")
    src = RAW_DIR / data_cfg["pourbaix"]["raw_subdir"]
    # Los archivos deben ser los registrados en la descarga (mismo SHA-256).
    bad = [n for n, h in pbx["files"].items() if sha256_file(src / "entries" / n) != h]
    if bad:
        raise RuntimeError(f"Entradas de Pourbaix distintas de las del MANIFEST: {bad[:5]}")
    n_jobs = int(cfg["execution"].get("n_jobs", 1))
    full_cfg = {**cfg, "pourbaix_download": {k: v for k, v in pbx.items() if k != "files"}}
    meta = build_run_metadata(full_cfg, [], n_jobs=n_jobs, allow_dirty=allow_dirty,
                              mp_database_version=pbx["mp_database_version"])  # fmt: skip
    meta["config_hash"] = config_hash({k: v for k, v in full_cfg.items() if k != "execution"})
    meta["status"] = "running"
    run_dir = make_run_dir(cfg["name"], meta["config_hash"])
    write_run_metadata(dict(meta), run_dir)

    systems = pd.read_csv(src / "systems.csv")
    tasks = [delayed(_composition_task)(r, src / "entries", cfg["conditions"])
             for r in systems.to_dict("records")]  # fmt: skip
    out = Parallel(n_jobs=n_jobs, backend="loky")(tasks)
    t_run = time.perf_counter()

    table = pd.DataFrame([r for o in out for r in o.get("rows", [])])
    excluded = pd.DataFrame([o["excluded"] for o in out if "excluded" in o])
    lt = {"lineterminator": "\n", "index": False}
    table.sort_values(["composition", "pH", "ion_conc_M"]).to_csv(run_dir / "pourbaix.csv", **lt)
    excluded.to_csv(run_dir / "excluded.csv", **lt)
    summary = summarize(table, float(cfg["reference"]["dg_threshold_eV_atom"]))
    summary.to_csv(run_dir / "summary.csv", **lt)
    (RESULTS_DIR / cfg["name"]).mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS_DIR / cfg["name"] / "summary.csv", **lt)

    meta["status"] = "completed"
    meta["n_tasks"] = len(tasks)
    meta["counts"] = {"compositions": len(systems),
                      "computed": int(table["composition"].nunique()) if len(table) else 0,
                      "excluded": len(excluded)}  # fmt: skip
    meta["timings"] = {"run_s": round(t_run - t0, 3), "total_s": round(time.perf_counter() - t0, 3)}
    write_run_metadata(meta, run_dir)
    append_runs_index(RESULTS_DIR / "runs_index.csv", cfg["name"], run_dir, meta)
    return run_dir
