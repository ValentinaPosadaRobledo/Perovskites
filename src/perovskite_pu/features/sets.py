"""Acceso único a las familias de features (`pre_dft`, `pre_dft_tau`, `post_dft`, ...).

Los experimentos (Fase 3) piden una familia por nombre y reciben solo sus columnas. Los
metadatos (asignación de Bartel, entrada de MP elegida, si es experimental...) nunca se
devuelven como features: están en archivos aparte.

Las familias `post_dft*` solo admiten composiciones elegibles (con polimorfo perovskita
calculado en esa funcional). Pedir otra composición es un error, no un NaN: así no se puede
imputar por accidente un hueco que depende de la etiqueta.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from functools import cache
from typing import Any

import pandas as pd

from perovskite_pu.utils.paths import CONFIGS_DIR, ROOT


@cache
def _output_paths() -> dict[str, str]:
    """Rutas de las tablas de features (`features.output` de configs/data.yaml)."""
    import yaml

    return yaml.safe_load((CONFIGS_DIR / "data.yaml").read_text("utf-8"))["features"]["output"]


def _registry() -> dict[str, Any]:
    """Familias disponibles: columnas, tabla(s) de origen y columna de elegibilidad."""
    out = _output_paths()
    pre = json.loads((ROOT / out["columns"]).read_text("utf-8"))
    reg = {fam: {"columns": cols, "post": False, "eligibility": None}
           for fam, cols in pre["families"].items()}  # fmt: skip
    post_path = ROOT / out["post_columns"]
    if post_path.exists():
        post = json.loads(post_path.read_text("utf-8"))
        for fam, cols in post["families"].items():
            reg[fam] = {"columns": cols, "post": True, "eligibility": post["eligibility"][fam]}
    return reg


def families() -> list[str]:
    """Nombres de las familias de features calculadas."""
    return sorted(_registry())


def eligible(family: str, compositions: Sequence[str]) -> pd.Series:
    """Serie booleana (índice = composición): True si la familia admite esa composición."""
    spec = _registry()[family]
    if spec["eligibility"] is None:
        return pd.Series(True, index=pd.Index(list(compositions)))
    meta = pd.read_parquet(ROOT / _output_paths()["post_metadata"])
    return meta[spec["eligibility"]].reindex(list(compositions)).fillna(False).astype(bool)


def load_feature_matrix(family: str, compositions: Sequence[str]) -> pd.DataFrame:
    """Features de una familia para las composiciones pedidas, en ese orden.

    Falla si la familia no existe, si falta alguna composición o si alguna no es elegible.
    """
    reg = _registry()
    if family not in reg:
        raise KeyError(f"Familia de features desconocida: {family} (hay: {sorted(reg)})")
    spec, out = reg[family], _output_paths()
    comps = list(compositions)
    table = pd.read_parquet(ROOT / out["table"])
    if spec["post"]:
        ok = eligible(family, comps)
        if not ok.all():
            bad = ok.index[~ok].tolist()
            raise ValueError(
                f"{len(bad)} composiciones no elegibles para {family} (p. ej. {bad[:5]}): "
                "filtrar antes con `eligible`, nunca imputar."
            )
        table = table.join(pd.read_parquet(ROOT / out["post_table"]))
    missing = sorted(set(comps) - set(table.index))
    if missing:
        extra = max(len(missing) - 5, 0)
        raise KeyError(f"Composiciones sin features: {missing[:5]} (y {extra} más)")
    return table.loc[comps, spec["columns"]]
