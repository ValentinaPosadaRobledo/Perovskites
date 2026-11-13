"""Gestión de semillas (regla 2.2 de CLAUDE.md).

Las semillas solo se definen en ``configs/seeds.yaml``. De cada semilla del experimento
se derivan tres flujos aleatorios independientes con ``SeedSequence.spawn(3)``:

1. ``fold_assignment``: asignación de composiciones a folds (solo esquemas aleatorios;
   en leave-one-group-out los folds son deterministas y este flujo no se usa).
2. ``pu_sampling``: muestreo de no etiquetados en cada bolsa del bagging PU.
3. ``model_init``: inicialización de los modelos (``random_state`` de sklearn).

Así, cambiar por ejemplo el número de bolsas no altera la asignación de folds.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from perovskite_pu.utils.paths import CONFIGS_DIR

# El orden importa: define qué hijo de spawn(3) corresponde a cada flujo.
STREAM_NAMES: tuple[str, str, str] = ("fold_assignment", "pu_sampling", "model_init")


@dataclass(frozen=True)
class SeedConfig:
    """Contenido validado de ``configs/seeds.yaml``."""

    seeds: tuple[int, ...]
    holdout_seed: int
    bootstrap_seed: int
    dev_n_seeds: int = 3

    def active_seeds(self, dev_mode: bool = False) -> tuple[int, ...]:
        """Semillas a usar: todas, o solo las primeras ``dev_n_seeds`` en modo desarrollo."""
        return self.seeds[: self.dev_n_seeds] if dev_mode else self.seeds


def load_seed_config(path: Path | str | None = None) -> SeedConfig:
    """Lee y valida el archivo de semillas.

    Raises:
        ValueError: si hay semillas repetidas, no enteras o negativas, o si
            ``holdout_seed`` o ``bootstrap_seed`` coinciden con otra semilla.
    """
    path = Path(path) if path is not None else CONFIGS_DIR / "seeds.yaml"
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    seeds = raw.get("seeds")
    holdout_seed = raw.get("holdout_seed")
    bootstrap_seed = raw.get("bootstrap_seed")
    dev_n_seeds = raw.get("dev_n_seeds", 3)

    # Validaciones: enteros no negativos y sin duplicados (bool es subclase de int: se excluye).
    if not seeds or not all(isinstance(s, int) and not isinstance(s, bool) for s in seeds):
        raise ValueError("`seeds` debe ser una lista no vacía de enteros.")
    if any(s < 0 for s in seeds):
        raise ValueError("Las semillas deben ser no negativas.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Hay semillas repetidas en `seeds`.")
    for name, value in (("holdout_seed", holdout_seed), ("bootstrap_seed", bootstrap_seed)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"`{name}` debe ser un entero.")
        if value in seeds:
            raise ValueError(f"`{name}` debe ser independiente de las semillas de experimento.")
    if holdout_seed == bootstrap_seed:
        raise ValueError("`holdout_seed` y `bootstrap_seed` deben ser distintas.")
    if not 1 <= dev_n_seeds <= len(seeds):
        raise ValueError("`dev_n_seeds` debe estar entre 1 y len(seeds).")

    return SeedConfig(
        seeds=tuple(seeds),
        holdout_seed=holdout_seed,
        bootstrap_seed=bootstrap_seed,
        dev_n_seeds=dev_n_seeds,
    )


def spawn_streams(seed: int) -> dict[str, np.random.SeedSequence]:
    """Deriva las tres ``SeedSequence`` hijas de una semilla, indexadas por nombre."""
    children = np.random.SeedSequence(seed).spawn(len(STREAM_NAMES))
    return dict(zip(STREAM_NAMES, children, strict=True))


def rng_streams(seed: int) -> dict[str, np.random.Generator]:
    """Igual que :func:`spawn_streams`, pero devuelve generadores ``np.random.Generator``."""
    return {name: np.random.default_rng(ss) for name, ss in spawn_streams(seed).items()}


def sklearn_random_state(seed_sequence: np.random.SeedSequence) -> int:
    """Convierte una ``SeedSequence`` en un entero válido para ``random_state`` de sklearn.

    sklearn exige un entero en [0, 2**32 - 1]; ``generate_state`` devuelve uint32.
    """
    return int(seed_sequence.generate_state(1, dtype=np.uint32)[0])
