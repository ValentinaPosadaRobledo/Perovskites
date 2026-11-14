"""Metadatos de corrida y control del árbol de git (regla 2.2 de CLAUDE.md).

Cada corrida guarda un ``run_metadata.json`` con todo lo necesario para reproducirla:
commit, configuración y su hash, semillas, versiones de paquetes, hash de ``uv.lock``,
versión de la base de datos de MP, CPU, ``n_jobs`` y tiempos.

Una corrida se NIEGA a ejecutarse con el árbol de git sucio, salvo con
``allow_dirty=True``; en ese caso el diff se guarda junto a los metadatos.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from collections.abc import Iterable
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from perovskite_pu.utils.hashing import config_hash, sha256_file
from perovskite_pu.utils.paths import ROOT

# Paquetes cuya versión se registra siempre.
TRACKED_PACKAGES: tuple[str, ...] = (
    "perovskite-pu",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "pymatgen",
    "mp-api",
    "emmet-core",
    "matminer",
    "shap",
)


class DirtyTreeError(RuntimeError):
    """El árbol de trabajo tiene cambios sin commit y no se pasó ``allow_dirty``."""


def _git(args: list[str], repo: Path) -> str:
    """Ejecuta un comando git en ``repo`` y devuelve su salida estándar."""
    result = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True)
    return result.stdout


def git_commit(repo: Path = ROOT) -> str:
    """Hash completo del commit actual (HEAD)."""
    return _git(["rev-parse", "HEAD"], repo).strip()


# Directorios de SALIDA: un archivo nuevo sin seguimiento aquí no es código y no ensucia el
# árbol (p. ej. los resultados de una corrida que todavía no se han versionado).
OUTPUT_DIRS: tuple[str, ...] = ("results/",)


def git_is_dirty(repo: Path = ROOT) -> bool:
    """True si hay cambios sin commit, incluidos archivos nuevos no ignorados.

    Excepción: los archivos NUEVOS sin seguimiento dentro de `OUTPUT_DIRS` no cuentan, porque
    son salidas y no pueden cambiar el comportamiento del código. Un archivo YA versionado de
    `results/` que se modifique sí cuenta.
    """
    lines = _git(["status", "--porcelain", "--untracked-files=all"], repo).splitlines()
    relevant = [
        ln for ln in lines
        if ln.strip() and not (ln.startswith("?? ") and ln[3:].startswith(OUTPUT_DIRS))
    ]  # fmt: skip
    return bool(relevant)


def git_diff(repo: Path = ROOT) -> str:
    """Diff de los cambios sin commit respecto a HEAD, más la lista de archivos sin seguimiento."""
    diff = _git(["diff", "HEAD"], repo)
    untracked = _git(["ls-files", "--others", "--exclude-standard"], repo)
    if untracked.strip():
        diff += "\n# Archivos sin seguimiento:\n" + untracked
    return diff


def ensure_clean_tree(repo: Path = ROOT, allow_dirty: bool = False) -> str | None:
    """Verifica que el árbol esté limpio.

    Returns:
        ``None`` si está limpio; el diff si está sucio y ``allow_dirty`` es True.

    Raises:
        DirtyTreeError: si está sucio y ``allow_dirty`` es False.
    """
    if not git_is_dirty(repo):
        return None
    if not allow_dirty:
        raise DirtyTreeError(
            "El árbol de git tiene cambios sin commit. Haz commit antes de ejecutar "
            "o usa --allow-dirty (el diff se guardará con la corrida)."
        )
    return git_diff(repo)


def package_versions(packages: Iterable[str] = TRACKED_PACKAGES) -> dict[str, str | None]:
    """Versión instalada de cada paquete (``None`` si no está instalado)."""
    versions: dict[str, str | None] = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def build_run_metadata(
    config: dict[str, Any],
    seeds: Iterable[int],
    *,
    n_jobs: int,
    mp_database_version: str | None = None,
    allow_dirty: bool = False,
    repo: Path = ROOT,
) -> dict[str, Any]:
    """Construye el diccionario de metadatos de una corrida.

    Llama a :func:`ensure_clean_tree`, así que falla con el árbol sucio salvo
    ``allow_dirty``. Los tiempos se añaden después con la clave ``timings``.
    """
    dirty_diff = ensure_clean_tree(repo, allow_dirty=allow_dirty)
    uv_lock = repo / "uv.lock"
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(repo),
        "git_dirty": dirty_diff is not None,
        "config": config,
        "config_hash": config_hash(config),
        "seeds": list(seeds),
        "package_versions": package_versions(),
        "uv_lock_sha256": sha256_file(uv_lock) if uv_lock.exists() else None,
        "mp_database_version": mp_database_version,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "n_jobs": n_jobs,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "timings": {},
        # El diff no va dentro del JSON (puede ser grande): se guarda en git_diff.patch.
        "_dirty_diff": dirty_diff,
    }


def make_run_dir(experiment: str, cfg_hash: str, results_root: Path | None = None) -> Path:
    """Crea ``results/<experimento>/<timestamp>_<hash_config>/raw/``.

    Devuelve el directorio de la corrida (el padre de ``raw/``).
    """
    results_root = results_root if results_root is not None else ROOT / "results"
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = results_root / experiment / f"{stamp}_{cfg_hash}"
    (run_dir / "raw").mkdir(parents=True, exist_ok=False)
    return run_dir


def write_run_metadata(meta: dict[str, Any], run_dir: Path) -> Path:
    """Escribe ``run_metadata.json`` (y ``git_diff.patch`` si el árbol estaba sucio)."""
    meta = dict(meta)
    diff = meta.pop("_dirty_diff", None)
    if diff:
        (run_dir / "git_diff.patch").write_text(diff, encoding="utf-8", newline="\n")
    out = run_dir / "run_metadata.json"
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False, sort_keys=True)
        fh.write("\n")
    return out


def append_runs_index(
    index_path: Path, experiment: str, run_dir: Path, meta: dict[str, Any], repo: Path = ROOT
) -> None:
    """Añade la corrida a ``results/runs_index.csv`` (lo crea con cabecera si no existe).

    ``run_dir`` puede ser relativo (p. ej. el que se pasa a ``--resume``): se resuelve antes
    de expresarlo respecto a la raíz del repositorio. Sin esto, una corrida reanudada con una
    ruta relativa fallaba al final, después de escribir todos los resultados (2026-11-14).
    """
    import pandas as pd

    row = pd.DataFrame([{
        "timestamp_utc": meta["timestamp_utc"], "experiment": experiment,
        "run_dir": Path(run_dir).resolve().relative_to(Path(repo).resolve()).as_posix(),
        "git_commit": meta["git_commit"], "git_dirty": meta["git_dirty"],
        "config_hash": meta["config_hash"], "n_seeds": len(meta["seeds"]),
        "n_jobs": meta["n_jobs"], "total_s": meta["timings"]["total_s"],
    }])  # fmt: skip
    header = not index_path.exists()
    row.to_csv(index_path, mode="a", header=header, index=False, lineterminator="\n")
