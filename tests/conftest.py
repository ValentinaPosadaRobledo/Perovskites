"""Fixtures compartidas por los tests.

Incluye:
- Datos sintéticos de juguete (``tests/fixtures/toy_abo3.csv``) y configuración real.
- Un audit hook (``sys.addaudithook``) que, mientras está armado, hace fallar cualquier
  apertura de ``holdout_frozen.csv`` (regla 2.1.9 de CLAUDE.md).
- La lógica de los tests dependientes de datos: se SALTAN si no hay datos, pero FALLAN
  si ``REQUIRE_DATA=1`` o si ``data/processed/`` ya contiene archivos.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from perovskite_pu.utils.paths import CONFIGS_DIR, HOLDOUT_FILENAME, PROCESSED_DIR

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# ---------------------------------------------------------------------------
# Audit hook contra la lectura del holdout
# ---------------------------------------------------------------------------
# Un audit hook no se puede desinstalar, así que se instala una sola vez y se activa
# o desactiva con esta bandera. Limitación conocida: solo intercepta aperturas hechas
# desde Python (open/io.open/os.open); una lectura en C pura (p. ej. pyarrow nativo)
# no dispara el evento. Por eso además existe el análisis estático.
_GUARD = {"armed": False}


class HoldoutAccessError(RuntimeError):
    """Se intentó abrir el holdout mientras el guardián estaba armado."""


def _holdout_audit_hook(event: str, args: tuple) -> None:
    if not _GUARD["armed"] or event != "open" or not args:
        return
    target = args[0]
    if isinstance(target, int):  # descriptor de archivo ya abierto: no hay ruta
        return
    try:
        name = os.fsdecode(target)
    except TypeError:
        return
    if os.path.basename(name) == HOLDOUT_FILENAME:
        raise HoldoutAccessError(f"Acceso prohibido al holdout durante el entrenamiento: {name}")


sys.addaudithook(_holdout_audit_hook)


@contextmanager
def holdout_guard() -> Iterator[None]:
    """Arma el guardián del holdout dentro del bloque ``with``."""
    _GUARD["armed"] = True
    try:
        yield
    finally:
        _GUARD["armed"] = False


@pytest.fixture
def guard_holdout():
    """Fixture que devuelve el context manager del guardián."""
    return holdout_guard


# ---------------------------------------------------------------------------
# Tests dependientes de datos: saltar o fallar
# ---------------------------------------------------------------------------
def real_data_present() -> bool:
    """True si ``data/processed/`` contiene algún archivo además de ``.gitkeep``.

    Interpretación documentada en la bitácora: el directorio existe siempre en el
    repositorio (con .gitkeep), así que lo que cuenta es que tenga datos.
    """
    if not PROCESSED_DIR.exists():
        return False
    return any(p.name != ".gitkeep" for p in PROCESSED_DIR.iterdir())


def data_required() -> bool:
    """True si la ausencia de datos debe ser un FALLO y no un salto."""
    return os.environ.get("REQUIRE_DATA") == "1" or real_data_present()


@pytest.fixture
def require_files():
    """Devuelve una función que salta (o falla) el test si faltan los archivos indicados."""

    def _require(*paths: Path) -> None:
        missing = [str(p) for p in paths if not Path(p).exists()]
        if not missing:
            return
        msg = f"Faltan datos: {missing}"
        if data_required():
            pytest.fail(msg + " (REQUIRE_DATA=1 o data/processed/ con datos: no se permite saltar)")
        # En CI se salta porque los datos (p. ej. train_pool) no se versionan; en local se
        # comprueba con `uv run poe test-data`, que exige los datos.
        pytest.skip(msg + " (datos no versionados; en local: `uv run poe test-data`)")

    return _require


# ---------------------------------------------------------------------------
# Datos sintéticos y configuración
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def toy_df() -> pd.DataFrame:
    """80 composiciones ABO3 sintéticas con etiqueta PU, año y features f0–f3."""
    return pd.read_csv(FIXTURES_DIR / "toy_abo3.csv")


@pytest.fixture(scope="session")
def toy_features(toy_df: pd.DataFrame) -> np.ndarray:
    """Matriz de features sin huecos (para el clustering del esquema `cluster`)."""
    return toy_df[["f0", "f1", "f2"]].to_numpy()


@pytest.fixture(scope="session")
def data_cfg() -> dict:
    """``configs/data.yaml`` real, para que los tests validen también la configuración."""
    with open(CONFIGS_DIR / "data.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="session")
def split_cfg(data_cfg: dict) -> dict:
    """Sección ``splits`` con valores adaptados al tamaño de los datos de juguete."""
    cfg = {k: dict(v) if isinstance(v, dict) else v for k, v in data_cfg["splits"].items()}
    cfg["cluster"]["n_clusters"] = 4  # 80 composiciones: 4 clústeres bastan
    cfg["temporal"]["cutoff_year"] = 2000  # fijo para los tests; la config real lo calcula
    return cfg


@pytest.fixture(scope="session")
def b_families(data_cfg: dict) -> dict:
    return data_cfg["b_families"]
