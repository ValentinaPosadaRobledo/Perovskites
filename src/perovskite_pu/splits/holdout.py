"""F1.8: holdout externo congelado, definido por grupos de elementos B (regla 2.1.5).

Selección (determinista con `holdout_seed`):
1. Candidatos: elementos B con al menos una positiva en el pool, agrupados por familia.
2. En cada familia, los candidatos se ordenan al azar con un generador derivado de
   `holdout_seed` y se añaden hasta cubrir `target` × (positivas de la familia).
3. Si la fracción total de positivas reservadas queda fuera de [mín, máx], se repite con el
   siguiente orden aleatorio del mismo generador (determinista). Se registra el intento.

El holdout son TODAS las composiciones del pool PU cuyo B está en el conjunto reservado.
`train_pool` es el resto. Ninguna composición puede estar en ambos.

Este módulo crea el holdout; solo `perovskite_pu.evaluation.holdout.load_holdout` lo LEE.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class HoldoutSelection:
    """Elementos B reservados y cifras de la selección."""

    reserved_b: list[str]
    attempt: int
    positive_fraction: float
    per_family: dict[str, list[str]] = field(default_factory=dict)


def select_reserved_b(
    pool: pd.DataFrame,
    holdout_seed: int,
    target: float = 0.125,
    bounds: Sequence[float] = (0.10, 0.15),
    max_attempts: int = 1000,
) -> HoldoutSelection:
    """Elige los elementos B reservados, estratificando por familia (ver docstring del módulo)."""
    pos = pool[pool["label"] == 1]
    total = len(pos)
    pos_by_b = pos.groupby("B").size()
    fam_of_b = pool.drop_duplicates("B").set_index("B")["family"]
    rng = np.random.default_rng(np.random.SeedSequence(holdout_seed))
    for attempt in range(1, max_attempts + 1):
        reserved, per_family = [], {}
        for fam in sorted(fam_of_b.unique()):
            cands = sorted(b for b in pos_by_b.index if fam_of_b.get(b) == fam)
            goal = target * pos_by_b[cands].sum() if cands else 0
            chosen, acc = [], 0
            for b in rng.permutation(cands):
                if acc >= goal:
                    break
                chosen.append(str(b))
                acc += int(pos_by_b[b])
            per_family[fam] = sorted(chosen)
            reserved += chosen
        frac = float(pos_by_b[reserved].sum() / total) if reserved else 0.0
        if bounds[0] <= frac <= bounds[1]:
            return HoldoutSelection(sorted(reserved), attempt, frac, per_family)
    raise RuntimeError(f"Sin selección en [{bounds[0]}, {bounds[1]}] tras {max_attempts} intentos.")


def split_holdout(
    pool: pd.DataFrame, reserved_b: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(holdout, train_pool): el holdout son todas las composiciones con B reservado."""
    mask = pool["B"].isin(set(reserved_b))
    holdout, train_pool = pool[mask].copy(), pool[~mask].copy()
    if set(holdout["composition"]) & set(train_pool["composition"]):
        raise AssertionError("Composiciones en holdout y train_pool a la vez.")
    return holdout.reset_index(drop=True), train_pool.reset_index(drop=True)
