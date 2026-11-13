"""Rutas canónicas del proyecto.

Todas las rutas se resuelven a partir de la raíz del repositorio para que el código
funcione igual sin importar el directorio desde el que se ejecute.
"""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Devuelve la raíz del repositorio.

    Se puede forzar con la variable de entorno ``PEROVSKITE_PU_ROOT`` (útil en tests
    o si el paquete se instala fuera del repositorio).
    """
    override = os.environ.get("PEROVSKITE_PU_ROOT")
    if override:
        return Path(override).resolve()
    # src/perovskite_pu/utils/paths.py -> parents[3] es la raíz del repositorio
    return Path(__file__).resolve().parents[3]


ROOT = project_root()
CONFIGS_DIR = ROOT / "configs"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
RESULTS_DIR = ROOT / "results"

# Nombres de archivo con contrato fijo (regla 2.1.5 de CLAUDE.md).
HOLDOUT_FILENAME = "holdout_frozen.csv"
HASHES_FILENAME = "HASHES.json"
TRAIN_POOL_FILENAME = "train_pool.parquet"
