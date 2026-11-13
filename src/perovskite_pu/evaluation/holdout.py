"""Acceso controlado al conjunto de prueba externo congelado (regla 2.1.5 de CLAUDE.md).

Este es el ÚNICO punto de lectura de ``data/splits/holdout_frozen.csv``. Exige la
variable de entorno ``HOLDOUT_UNLOCK=1``, que solo se activa en la evaluación final
(F4.1), después de congelar y depositar el preregistro en OSF.

Ningún módulo de ``perovskite_pu.training`` puede importar este módulo; lo verifica
un test estático en ``tests/test_leakage.py``.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import pandas as pd

from perovskite_pu.utils.hashing import read_hashes, sha256_bytes
from perovskite_pu.utils.paths import HASHES_FILENAME, HOLDOUT_FILENAME, SPLITS_DIR

UNLOCK_ENV_VAR = "HOLDOUT_UNLOCK"


class HoldoutLockedError(PermissionError):
    """Se intentó leer el conjunto reservado sin ``HOLDOUT_UNLOCK=1``."""


class HoldoutIntegrityError(RuntimeError):
    """El archivo reservado no coincide con el hash registrado en HASHES.json."""


def load_holdout(
    path: Path | str | None = None, hashes_path: Path | str | None = None
) -> pd.DataFrame:
    """Lee el holdout congelado tras verificar el desbloqueo y la integridad del archivo.

    El archivo se lee UNA vez como bytes: el hash se calcula sobre esos mismos bytes y
    luego se parsean, así no hay ventana entre la verificación y la lectura.

    Raises:
        HoldoutLockedError: si ``HOLDOUT_UNLOCK`` no vale "1".
        HoldoutIntegrityError: si no hay hash registrado o no coincide.
    """
    if os.environ.get(UNLOCK_ENV_VAR) != "1":
        raise HoldoutLockedError(
            f"El holdout está bloqueado. Solo se abre en F4.1 con {UNLOCK_ENV_VAR}=1, "
            "después de congelar el preregistro."
        )
    path = Path(path) if path is not None else SPLITS_DIR / HOLDOUT_FILENAME
    hashes_path = Path(hashes_path) if hashes_path is not None else SPLITS_DIR / HASHES_FILENAME

    expected = read_hashes(hashes_path).get(path.name)
    if expected is None:
        raise HoldoutIntegrityError(f"No hay hash registrado para {path.name} en {hashes_path}.")
    data = path.read_bytes()
    actual = sha256_bytes(data)
    if actual != expected:
        raise HoldoutIntegrityError(
            f"Hash de {path.name} distinto del registrado: {actual} != {expected}."
        )
    return pd.read_csv(io.BytesIO(data))
