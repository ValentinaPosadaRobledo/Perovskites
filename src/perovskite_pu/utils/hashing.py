"""Hashing canónico de archivos y configuraciones.

La forma canónica de un CSV es: filas ordenadas por las columnas clave, columnas en
orden fijo, codificación UTF-8 y fin de línea LF. Así el SHA-256 no depende del sistema
operativo (CRLF en Windows) ni del orden en que se generaron las filas.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

_CHUNK = 1 << 20  # 1 MiB por lectura para no cargar archivos grandes en memoria


def sha256_bytes(data: bytes) -> str:
    """SHA-256 hexadecimal de un bloque de bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path | str) -> str:
    """SHA-256 hexadecimal de un archivo, leído en modo binario (sin traducir fines de línea)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def canonical_csv_bytes(df: pd.DataFrame, sort_by: Sequence[str]) -> bytes:
    """Serializa un DataFrame en forma canónica (ver docstring del módulo).

    Args:
        df: tabla a serializar. Las columnas se ordenan alfabéticamente, excepto las de
            ``sort_by``, que van primero y en ese orden.
        sort_by: columnas clave para ordenar las filas (p. ej. ``["composition"]``).
    """
    missing = [c for c in sort_by if c not in df.columns]
    if missing:
        raise KeyError(f"Columnas de orden ausentes: {missing}")
    columns = list(sort_by) + sorted(c for c in df.columns if c not in sort_by)
    ordered = df[columns].sort_values(list(sort_by), kind="mergesort").reset_index(drop=True)
    # lineterminator="\n" fuerza LF también en Windows.
    text = ordered.to_csv(index=False, lineterminator="\n")
    return text.encode("utf-8")


def write_canonical_csv(df: pd.DataFrame, path: Path | str, sort_by: Sequence[str]) -> str:
    """Escribe el CSV canónico en ``path`` y devuelve su SHA-256.

    Se escribe en modo binario para que Python no traduzca ``\\n`` a ``\\r\\n``.
    """
    data = canonical_csv_bytes(df, sort_by)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return sha256_bytes(data)


def config_hash(config: dict[str, Any], length: int = 12) -> str:
    """Hash corto y estable de una configuración (independiente del orden de las claves)."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_bytes(canonical.encode("utf-8"))[:length]


def read_hashes(path: Path | str) -> dict[str, str]:
    """Lee un archivo HASHES.json ({nombre_de_archivo: sha256}). Devuelve {} si no existe."""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        content = json.load(fh)
    return dict(content.get("files", {}))
