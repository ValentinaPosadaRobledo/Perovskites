"""F5.1: descarga de las entradas de Pourbaix de Materials Project por sistema A–B–O–H.

Uso (desde la raíz del repositorio):

    uv run python -m perovskite_pu.pourbaix.download            # usa configs/data.yaml
    uv run python -m perovskite_pu.pourbaix.download --dry-run  # solo cuenta los sistemas

Qué hace:

1. Elige los sistemas: las composiciones de `train_pool` con polimorfo perovskita GGA/GGA+U
   (las elegibles de `post_dft`; decisión de la autora, 2026-11-20). El conjunto reservado no
   se consulta.
2. Descarga una vez la tabla de iones de referencia (proyecto `ion_ref_data` de MPContribs)
   por su API REST pública y se la pasa a mp-api, que si no exigiría el cliente de MPContribs.
3. Para cada sistema, `MPRester.get_pourbaix_entries([A, B])` (GGA/GGA+U con
   MaterialsProject2020Compatibility y MaterialsProjectAqueousCompatibility). Guarda las
   entradas en `data/raw/pourbaix/entries/<A>-<B>.json.gz`, una a una y de forma atómica: la
   descarga se puede reanudar.
4. Registra en la sección `pourbaix` de `data/MANIFEST.json` la versión de la base de datos, las
   versiones de los paquetes, la fecha, el commit y el SHA-256 de cada archivo.

La clave de MP no se maneja aquí: mp-api la lee de `MP_API_KEY` (en la nube, el proxy la
inyecta y basta un valor ficticio de 32 caracteres).
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from perovskite_pu.utils.hashing import sha256_file
from perovskite_pu.utils.paths import CONFIGS_DIR, DATA_DIR, RAW_DIR, ROOT
from perovskite_pu.utils.run_metadata import git_commit, git_is_dirty

MANIFEST_PATH = DATA_DIR / "MANIFEST.json"
ION_FILE = "ion_ref_data.json"


def candidate_systems(pool: pd.DataFrame, post_meta: pd.DataFrame) -> pd.DataFrame:
    """Composiciones de `train_pool` con polimorfo perovskita GGA/GGA+U y su sistema A–B.

    `post_meta` puede contener composiciones fuera de `train_pool` (p. ej. las reservadas):
    solo se toman las de `pool`.
    """
    elig = post_meta.index[post_meta["eligible_post_dft"].astype(bool)]
    sel = pool[pool["composition"].isin(elig)].copy()
    sel["material_id_gga"] = sel["composition"].map(post_meta["material_id_gga"])
    sel["system"] = sel["A"] + "-" + sel["B"]
    return sel[["composition", "A", "B", "label", "material_id_gga", "system"]].sort_values(
        "composition", ignore_index=True
    )


def fetch_ion_reference(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Tabla de iones de referencia desde la API REST pública de MPContribs.

    Raises:
        RuntimeError: si la respuesta viene paginada o incompleta.
    """
    import requests

    params = {"project": cfg["project"], "_fields": ",".join(cfg["fields"]),
              "_limit": int(cfg["limit"])}  # fmt: skip
    resp = requests.get(cfg["url"], params=params, timeout=120)
    resp.raise_for_status()
    body = resp.json()
    data = body["data"]
    if body.get("has_more") or len(data) != body.get("total_count"):
        raise RuntimeError("La tabla de iones viene paginada o incompleta: subir `limit`.")
    # mp-api usa estas claves; si falta alguna, el diagrama sería incorrecto en silencio.
    need = {"charge", "MajElements", "RefSolid", "ΔGᶠ", "ΔGᶠRefSolid"}
    if any(need - set(d["data"]) for d in data):
        raise RuntimeError("Registros de iones sin todos los campos que usa mp-api.")
    return data


def _write_atomic(path: Path, payload: Any) -> None:
    """JSON comprimido (gzip sin fecha) escrito en un temporal y renombrado."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as fh:
        fh.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    os.replace(tmp, path)


def download(config_path: Path, dry_run: bool = False, limit: int | None = None) -> dict:
    """Descarga (o reanuda) las entradas de Pourbaix de todos los sistemas candidatos."""
    from monty.json import jsanitize

    cfg = yaml.safe_load(Path(config_path).read_text("utf-8"))
    pcfg = cfg["pourbaix"]
    out = RAW_DIR / pcfg["raw_subdir"]
    pool = pd.read_parquet(ROOT / cfg["holdout"]["train_pool_path"])
    post_meta = pd.read_parquet(ROOT / cfg["features"]["output"]["post_metadata"])
    cands = candidate_systems(pool, post_meta)
    systems = sorted(cands["system"].unique())
    if limit is not None:
        systems = systems[:limit]
    if dry_run:
        return {"compositions": len(cands), "systems": len(systems)}

    (out / "entries").mkdir(parents=True, exist_ok=True)
    ion_path = out / ION_FILE
    if not ion_path.exists():
        ion_path.write_text(json.dumps(fetch_ion_reference(pcfg["ion_reference"]),
                                       ensure_ascii=False, indent=1) + "\n", "utf-8")  # fmt: skip
    ion = json.loads(ion_path.read_text("utf-8"))
    cands.to_csv(out / "systems.csv", index=False, lineterminator="\n")

    from mp_api.client import MPRester

    t0, done = time.perf_counter(), 0
    with MPRester(mute_progress_bars=True) as mpr:
        mpr.get_ion_reference_data = lambda: ion  # evita el cliente de MPContribs
        db_version = mpr.get_database_version()
        for k, system in enumerate(systems, 1):
            path = out / "entries" / f"{system}.json.gz"
            if path.exists():  # reanudable
                continue
            entries = mpr.get_pourbaix_entries(system.split("-"),
                                               solid_compat=pcfg["solid_compat"])  # fmt: skip
            _write_atomic(path, jsanitize([e.as_dict() for e in entries], strict=True))
            done += 1
            if done % 25 == 0:
                rate = (time.perf_counter() - t0) / done
                print(f"  {k}/{len(systems)} · faltan ~{rate * (len(systems) - k) / 60:.0f} min",
                      flush=True)  # fmt: skip

    files = sorted((out / "entries").glob("*.json.gz"))
    manifest = json.loads(MANIFEST_PATH.read_text("utf-8")) if MANIFEST_PATH.exists() else {}
    manifest["pourbaix"] = {
        "dir": f"data/raw/{pcfg['raw_subdir']}", "mp_database_version": db_version,
        "mp_api_version": metadata.version("mp-api"),
        "emmet_core_version": metadata.version("emmet-core"),
        "pymatgen_version": metadata.version("pymatgen"),
        "query_date_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(), "git_dirty": git_is_dirty(),
        "solid_compat": pcfg["solid_compat"], "thermo_types": pcfg["thermo_types"],
        "ion_reference": {**pcfg["ion_reference"], "file": ION_FILE, "n_records": len(ion),
                          "sha256": sha256_file(ion_path)},
        "systems_file_sha256": sha256_file(out / "systems.csv"),
        "n_systems": len(files),
        "files": {p.name: sha256_file(p) for p in files},
    }  # fmt: skip
    with open(MANIFEST_PATH, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return {"systems": len(systems), "downloaded_now": done, "files": len(files)}


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=CONFIGS_DIR / "data.yaml")
    ap.add_argument("--dry-run", action="store_true", help="solo cuenta los sistemas")
    ap.add_argument("--limit", type=int, default=None, help="solo los N primeros (pruebas)")
    args = ap.parse_args(argv)
    print(json.dumps(download(args.config, args.dry_run, args.limit), indent=2))


if __name__ == "__main__":  # requisito en Windows (regla 2.2)
    main()
