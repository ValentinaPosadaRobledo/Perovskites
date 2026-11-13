"""F1.1: descarga de óxidos ABO3 desde Materials Project (MP).

Uso (desde la raíz del repositorio):

    uv run python -m perovskite_pu.data.download            # usa configs/data.yaml
    uv run python -m perovskite_pu.data.download --dry-run  # solo cuenta, no escribe

Qué hace:

1. Consulta el endpoint `summary` con la fórmula anónima ABC3 y O entre los elementos,
   excluyendo las entradas obsoletas (`deprecated`).
2. Repite la consulta con y sin GNoME para marcar con `is_gnome` las entradas que solo
   aparecen con GNoME (`gnome_policy: exclude_and_flag`, decisión de la autora).
3. Descarta las ABC3 en las que el O no es el elemento con subíndice 3.
4. Pide al endpoint `thermo` la E_hull y la ΔHf con UNA funcional explícita (GGA_GGA+U), y
   aparte r2SCAN para la sensibilidad. La E_hull de `summary` usa un esquema mixto y no se
   usa en el análisis.
5. Guarda en `data/raw/` (no versionado) y registra en `data/MANIFEST.json` la versión de la
   base de datos de MP, las versiones de mp-api/emmet-core/pymatgen, la fecha, los
   parámetros de la consulta y el SHA-256 de cada archivo.

La clave de MP NO se maneja aquí: mp-api la lee de `MP_API_KEY`. En la nube de Claude Code
el proxy la inyecta en la cabecera, y basta con un valor ficticio de 32 caracteres.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pymatgen.core import Composition

from perovskite_pu.utils.hashing import sha256_file
from perovskite_pu.utils.paths import CONFIGS_DIR, DATA_DIR, RAW_DIR
from perovskite_pu.utils.run_metadata import git_commit, git_is_dirty

SUMMARY_FILE = "mp_abo3_summary.json.gz"  # documentos completos, con estructuras
INDEX_FILE = "mp_abo3_index.parquet"  # tabla plana sin estructuras
THERMO_FILE = "mp_abo3_thermo.parquet"  # E_hull y ΔHf por thermo_type
MANIFEST_PATH = DATA_DIR / "MANIFEST.json"


# ---------------------------------------------------------------------------
# Funciones puras (probadas sin red en tests/test_download.py)
# ---------------------------------------------------------------------------
def is_abo3_oxide(formula: str) -> bool:
    """True si la composición reducida es A1 B1 O3 con A ≠ B ≠ O.

    La fórmula anónima ABC3 también admite casos como A3BO (O con subíndice 1); esos se
    descartan. Polianiones como KNO3 SÍ pasan este filtro: se excluyen en F1.3, porque ahí
    se decide con estados de oxidación y la configuración de exclusión.
    """
    try:
        comp = Composition(formula).reduced_composition
    except ValueError:
        return False
    amounts = {el.symbol: amt for el, amt in comp.items()}
    if len(amounts) != 3 or amounts.get("O") != 3:
        return False
    return all(amt == 1 for el, amt in amounts.items() if el != "O")


def flag_gnome(
    docs_with_gnome: Iterable[dict[str, Any]], ids_without_gnome: set[str]
) -> list[dict[str, Any]]:
    """Marca con `is_gnome` las entradas de GNoME, con dos criterios independientes.

    1. La entrada solo aparece en la consulta que incluye GNoME.
    2. Su lote de cálculo (`builder_meta.batch_id`) contiene "gnome"; mp-api filtra GNoME
       con `batch_id_not_eq=gnome_r2scan_statics`.
    """
    out = []
    for doc in docs_with_gnome:
        doc = dict(doc)
        batch = str((doc.get("builder_meta") or {}).get("batch_id") or "")
        only_in_gnome_query = str(doc["material_id"]) not in ids_without_gnome
        doc["is_gnome"] = only_in_gnome_query or "gnome" in batch.lower()
        out.append(doc)
    return out


def index_table(docs: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Tabla plana de los documentos (sin estructuras), una fila por material_id.

    `icsd_ids` se extrae de `database_IDs`. La etiqueta de positivo NO se asigna aquí: se
    decide en F1.5, contrastando `theoretical` con la presencia de IDs de ICSD.
    """
    rows = []
    for d in docs:
        db_ids = d.get("database_IDs") or {}
        icsd = db_ids.get("icsd") or []
        sym = d.get("symmetry") or {}
        rows.append(
            {
                "material_id": str(d["material_id"]),
                "formula_pretty": d.get("formula_pretty"),
                "reduced_formula": Composition(d["formula_pretty"]).reduced_formula,
                "theoretical": d.get("theoretical"),
                "n_icsd_ids": len(icsd),
                "icsd_ids": ";".join(str(i) for i in icsd),
                "spacegroup_number": sym.get("number"),
                "spacegroup_symbol": sym.get("symbol"),
                "crystal_system": str(sym.get("crystal_system") or ""),
                "summary_energy_above_hull": d.get("energy_above_hull"),  # esquema mixto
                "batch_id": (d.get("builder_meta") or {}).get("batch_id"),  # lote de cálculo
                "is_gnome": bool(d.get("is_gnome", False)),
            }
        )
    return pd.DataFrame(rows).sort_values("material_id", ignore_index=True)


def build_manifest(
    files: Sequence[Path],
    db_version: str,
    query: dict[str, Any],
    counts: dict[str, int],
    previous: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construye `data/MANIFEST.json` y conserva la descripción del manifiesto anterior."""
    versions = {}
    for pkg in ("mp-api", "emmet-core", "pymatgen"):
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    try:  # commit del código que hizo la descarga (y si el árbol estaba sucio)
        code = {"git_commit": git_commit(), "git_dirty": git_is_dirty()}
    except Exception:  # fuera de un repositorio git (p. ej. en tests aislados)
        code = {"git_commit": None, "git_dirty": None}
    return {
        "description": (previous or {}).get("description", "Registro de los datos crudos."),
        **code,
        "mp_database_version": db_version,
        "mp_api_version": versions["mp-api"],
        "emmet_core_version": versions["emmet-core"],
        "pymatgen_version": versions["pymatgen"],
        "query_date_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "query": query,
        "counts": counts,
        "files": {f.name: sha256_file(f) for f in files},
    }


# ---------------------------------------------------------------------------
# Consultas a la API (requieren red y clave)
# ---------------------------------------------------------------------------
def _summary_docs(mpr: Any, mp_cfg: dict[str, Any], include_gnome: bool) -> list[dict]:
    """Consulta el endpoint `summary` con la fórmula anónima y devuelve dicts."""
    kwargs: dict[str, Any] = {
        "formula": mp_cfg["formula_anonymous"],
        "elements": mp_cfg["elements_include"],
        "fields": mp_cfg["fields"],
        "include_gnome": include_gnome,
    }
    if mp_cfg.get("exclude_deprecated", True):
        kwargs["deprecated"] = False
    return list(mpr.materials.summary.search(**kwargs))


def _thermo_docs(
    mpr: Any, material_ids: Sequence[str], thermo_types: Sequence[str], chunk: int
) -> list[dict]:
    """E_hull y ΔHf para unos material_ids con thermo_types explícitos, en bloques."""
    fields = ["material_id", "thermo_type", "energy_above_hull", "formation_energy_per_atom"]
    out: list[dict] = []
    for start in range(0, len(material_ids), chunk):
        block = list(material_ids[start : start + chunk])
        out.extend(
            mpr.materials.thermo.search(
                material_ids=block, thermo_types=list(thermo_types), fields=fields
            )
        )
    return out


def download(config_path: Path, out_dir: Path, dry_run: bool = False) -> dict[str, int]:
    """Ejecuta la descarga completa de F1.1 y devuelve los conteos."""
    from mp_api.client import MPRester  # importación tardía: los tests no necesitan red

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))["materials_project"]
    policy = cfg.get("gnome_policy", "exclude_and_flag")
    with MPRester(use_document_model=False, mute_progress_bars=True) as mpr:
        db_version = mpr.db_version
        base = _summary_docs(mpr, cfg, include_gnome=False)
        base_ids = {str(d["material_id"]) for d in base}
        if policy in ("exclude_and_flag", "include"):
            # Ambas políticas descargan y marcan igual; la política solo decide después
            # (F1.5) si las filas `is_gnome` entran al análisis principal.
            docs = flag_gnome(_summary_docs(mpr, cfg, include_gnome=True), base_ids)
        elif policy == "exclude":
            docs = flag_gnome(base, base_ids)
        else:
            raise ValueError(f"gnome_policy desconocida: {policy!r}")

        n_abc3 = len(docs)
        docs = [d for d in docs if is_abo3_oxide(d["formula_pretty"])]
        ids = [str(d["material_id"]) for d in docs]
        types = list(cfg["thermo_types"]) + list(cfg.get("sensitivity_thermo_types", []))
        thermo = _thermo_docs(mpr, ids, types, int(cfg.get("thermo_chunk_size", 500)))

    index = index_table(docs)
    counts = {
        "abc3_oxide_entries": n_abc3,
        "abo3_entries": len(docs),
        "abo3_gnome_entries": int(index["is_gnome"].sum()),
        "abo3_reduced_formulas": int(index["reduced_formula"].nunique()),
        "abo3_with_icsd_id": int((index["n_icsd_ids"] > 0).sum()),
        "abo3_theoretical_false": int((index["theoretical"] == False).sum()),  # noqa: E712
        "thermo_docs": len(thermo),
    }
    if dry_run:
        return counts

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / SUMMARY_FILE
    # mtime=0 en gzip: el mismo contenido produce el mismo SHA-256.
    with open(summary_path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        gz.write(json.dumps(docs, sort_keys=True, default=str).encode("utf-8"))
    index_path = out_dir / INDEX_FILE
    index.to_parquet(index_path, index=False)
    thermo_path = out_dir / THERMO_FILE
    thermo_df = pd.DataFrame(thermo)
    thermo_df["material_id"] = thermo_df["material_id"].astype(str)
    thermo_df["thermo_type"] = thermo_df["thermo_type"].astype(str)
    thermo_df.sort_values(["material_id", "thermo_type"], ignore_index=True).to_parquet(
        thermo_path, index=False
    )

    previous = json.loads(MANIFEST_PATH.read_text("utf-8")) if MANIFEST_PATH.exists() else None
    query = {k: cfg[k] for k in cfg if k != "api_key_env"}
    manifest = build_manifest(
        [summary_path, index_path, thermo_path], db_version, query, counts, previous
    )
    with open(MANIFEST_PATH, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return counts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="F1.1: descarga de óxidos ABO3 desde MP.")
    parser.add_argument("--config", type=Path, default=CONFIGS_DIR / "data.yaml")
    parser.add_argument("--out", type=Path, default=RAW_DIR)
    parser.add_argument("--dry-run", action="store_true", help="Solo cuenta; no escribe nada.")
    args = parser.parse_args(argv)
    counts = download(args.config, args.out, dry_run=args.dry_run)
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":  # requisito en Windows (regla 2.2)
    main()
