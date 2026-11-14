"""Construcción del pool PU y generación de los folds de todos los esquemas (F1.7–F1.8).

El pool tiene una fila por composición con etiqueta `positive` (label = 1) o `unlabeled`
(label = 0). Las sintetizadas no perovskita, las ambiguas y los polianiones NO entran.
Columnas: composition, A, B, A_size, B_size, label, year, family y las banderas de F1.5.

`year` es el año de primera caracterización de las positivas: el más antiguo de las
referencias de ICSD de sus entradas perovskita experimentales (procedencia de MP). Las no
etiquetadas no tienen año.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from perovskite_pu.splits.schemes import SCHEMES, assign_b_family, make_folds, scheme_groups
from perovskite_pu.utils.seeds import rng_streams


def build_pool(
    labels: pd.DataFrame,
    entries: pd.DataFrame,
    provenance: pd.DataFrame,
    b_families: Mapping[str, Any],
) -> pd.DataFrame:
    """Pool PU (P ∪ U) con sitios, familia del elemento B, año y banderas."""
    pe = entries[(entries["cls_final"] == "perovskite") & entries["experimental"]]
    years = (
        pe.merge(provenance, on="material_id", how="left")
        .groupby("reduced_formula")["first_year"]
        .min()
    )
    pool = labels[labels["label"].isin(["positive", "unlabeled"])].copy()
    pool = pool.rename(columns={"reduced_formula": "composition"})
    pool["label"] = (pool["label"] == "positive").astype(int)
    pool["year"] = pool["composition"].map(years).where(pool["label"] == 1)
    pool["family"] = assign_b_family(pool["B"], b_families)
    return pool.sort_values("composition", ignore_index=True)


def generate_folds(
    pool: pd.DataFrame,
    features: np.ndarray,
    split_cfg: Mapping[str, Any],
    b_families: Mapping[str, Any],
    seeds: Sequence[int],
    schemes: Sequence[str] = SCHEMES,
) -> pd.DataFrame:
    """Tabla larga (scheme, seed, fold, composition, group).

    Cada fila indica que esa composición está en el TEST de ese fold.

    Los esquemas deterministas (loeo_A, loeo_B, family) dan los mismos folds con cualquier
    semilla; se generan igualmente para todas, para que cada corrida lea su propia partición.
    """
    rows = []
    for seed in seeds:
        for scheme in schemes:
            rng = rng_streams(seed)["fold_assignment"]
            folds = make_folds(pool, scheme, split_cfg, b_families, rng=rng, features=features)
            # Grupos recalculados con un generador idéntico (igual que dentro de make_folds).
            rng_g = rng_streams(seed)["fold_assignment"]
            groups = scheme_groups(
                pool, scheme, split_cfg, b_families, rng=rng_g, features=features
            )
            for k, fold in enumerate(folds):
                for i in fold.test_idx:
                    group = None if groups is None else groups.iloc[i]
                    comp = pool.at[i, "composition"]
                    rows.append(
                        {
                            "scheme": scheme,
                            "seed": seed,
                            "fold": k,
                            "composition": comp,
                            "group": group,
                        }
                    )
    return pd.DataFrame(rows)
