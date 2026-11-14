"""F2.3: auditoría de features (regla 2.1.8).

Comprueba, a partir de `configs/feature_audit.yaml`:
1. Que toda columna de toda familia pertenece a un grupo auditado (lista blanca).
2. Que ninguna columna coincide con un patrón prohibido (metadatos, procedencia, etiqueta).
3. Que la familia principal `pre_dft` solo contiene grupos anteriores a la síntesis y sin
   dependencia de la etiqueta.

Además ofrece un cribado univariante DESCRIPTIVO: el AUC de cada columna para separar P de U.
Una columna casi perfecta se revisa como posible fuga. El cribado nunca se usa para elegir
features (eso sería usar las etiquetas de todo el conjunto).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


def load_audit_config() -> dict[str, Any]:
    """Lee `configs/feature_audit.yaml`."""
    import yaml

    from perovskite_pu.utils.paths import CONFIGS_DIR

    return yaml.safe_load((CONFIGS_DIR / "feature_audit.yaml").read_text("utf-8"))


def audited_columns(
    audit_groups: Mapping[str, Any], columns_by_group: Mapping[str, Sequence[str]]
) -> dict[str, str]:
    """{columna: grupo auditado}, resolviendo `columns_from` con el JSON de columnas."""
    out = {}
    for group, spec in audit_groups.items():
        if "columns_from" in spec:
            cols = columns_by_group.get(spec["columns_from"], [])
        else:
            cols = spec.get("columns", [])
        for c in cols:
            out[c] = group
    return out


def check_family_columns(
    families: Mapping[str, Sequence[str]],
    audit_cfg: Mapping[str, Any],
    columns_by_group: Mapping[str, Sequence[str]],
) -> list[str]:
    """Lista de problemas encontrados (vacía si todo está en regla)."""
    problems = []
    col2group = audited_columns(audit_cfg["groups"], columns_by_group)
    patterns = [re.compile(p) for p in audit_cfg["forbidden_patterns"]]
    for fam, cols in families.items():
        if fam not in audit_cfg.get("families", {}):
            problems.append(f"{fam}: familia sin entrada en feature_audit.yaml")
        for c in cols:
            if any(p.search(c) for p in patterns):
                problems.append(f"{fam}: columna prohibida '{c}'")
            elif c not in col2group:
                problems.append(f"{fam}: columna sin grupo auditado '{c}'")
        if fam == "pre_dft":
            for c in cols:
                spec = audit_cfg["groups"].get(col2group.get(c, ""), {})
                if spec and (
                    spec["stage"] != "pre_synthesis" or spec["label_dependence"] != "none"
                ):
                    problems.append(f"pre_dft: '{c}' ({col2group[c]}) no es pre_synthesis sin "
                                    "dependencia de la etiqueta")  # fmt: skip
    return problems


def univariate_auc(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """AUC de cada columna para separar P (y = 1) de U (y = 0), ignorando NaN.

    Descriptivo: U no es negativo, así que es una cota inferior de la separación real. Devuelve
    también la fracción de NaN por clase (la ausencia puede separar por sí sola).
    """
    from sklearn.metrics import roc_auc_score

    y = pd.Series(np.asarray(y), index=X.index)
    rows = []
    for c in X.columns:
        v = X[c]
        ok = v.notna()
        yy, vv = y[ok], v[ok]
        # Sin las dos clases o con la columna constante, el AUC no está definido.
        degenerate = yy.nunique() < 2 or vv.nunique() < 2
        auc = np.nan if degenerate else float(roc_auc_score(yy, vv))
        rows.append({
            "feature": c, "n": int(ok.sum()), "auc": auc,
            "separation": abs(auc - 0.5) if not np.isnan(auc) else np.nan,
            "nan_frac_P": float(v[y == 1].isna().mean()),
            "nan_frac_U": float(v[y == 0].isna().mean()),
        })  # fmt: skip
    return pd.DataFrame(rows).sort_values("separation", ascending=False, ignore_index=True)


def suspicious(screen: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Columnas con AUC ≥ umbral o ≤ 1 − umbral."""
    a = screen["auc"]
    return screen[(a >= threshold) | (a <= 1 - threshold)]
