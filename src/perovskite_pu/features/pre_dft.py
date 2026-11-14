"""F2.1: features `pre_dft` por composición (composición + factores geométricos).

Familias (definidas en `configs/data.yaml`, sección `features.families`):
- `pre_dft` (principal): el conjunto MAGPIE completo de Ward et al. (2016), es decir,
  `ElementProperty("magpie")`, `Stoichiometry`, `ValenceOrbital` e `IonProperty` de matminer,
  más el factor de Goldschmidt t y el factor octaédrico μ. **Sin τ.**
- `pre_dft_tau` (sensibilidad 14 del preregistro): `pre_dft` + τ de Bartel et al. (2019).
  τ se ajustó con su TableS1, que contiene la mayoría de nuestras positivas. Por eso se evalúa
  en todo el conjunto (declarada in-sample) y solo sobre las positivas fuera de la TableS1
  (regla 2.1.6).

Por qué se calculan aquí y no dentro del Pipeline: cada feature es una transformación FIJA de
la fórmula, sin parámetros ajustados con datos, así que no hay fuga por calcularla una vez para
todas las composiciones. Lo que sí se ajusta (imputación, umbral de varianza, escalado) va en
el Pipeline de cada fold.

t, μ y τ usan la asignación de sitios y de estados de oxidación de Bartel, que solo depende de
la fórmula. Las columnas A/B del pool NO se usan: salen de la estructura y, en las positivas,
de sus entradas perovskita experimentales, así que dependen de la etiqueta.

Valores faltantes: t, μ y τ quedan como NaN cuando el algoritmo de Bartel no los define (sin
combinación de oxidación que equilibre la carga, radio no positivo o r_A ≤ r_B). No se imputan
aquí: se imputan dentro del Pipeline, con el train de cada fold.

Salidas (rutas en `features.output`):
- tabla de features (una fila por composición, índice `reduced_formula`);
- metadatos (asignación de Bartel, NC usados, notas y pertenencia a la TableS1);
- JSON con las columnas de cada grupo y de cada familia.

Uso:
    uv run python -m perovskite_pu.features.pre_dft
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import pandas as pd
from pymatgen.core import Composition

from perovskite_pu.features.tolerance import ShannonDict, default_tables, tau_descriptors
from perovskite_pu.utils.paths import CONFIGS_DIR, INTERIM_DIR, ROOT

# Columnas de los factores geométricos (prefijo `geom_` para distinguirlas de las de matminer).
GEOMETRIC_COLUMNS = {"t": "geom_t", "mu": "geom_mu", "tau": "geom_tau"}


def _featurizers() -> dict[str, Callable[[], Any]]:
    """Constructores de los featurizadores de composición (import tardío: matminer es lento)."""
    from matminer.featurizers.composition import (
        ElementProperty,
        IonProperty,
        Stoichiometry,
        ValenceOrbital,
    )

    return {
        "element_property_magpie": lambda: ElementProperty.from_preset("magpie", impute_nan=True),
        "stoichiometry": lambda: Stoichiometry(),
        "valence_orbital": lambda: ValenceOrbital(impute_nan=True),
        "ion_property": lambda: IonProperty(impute_nan=True),
    }


def composition_features(
    formulas: Iterable[str], groups: Sequence[str]
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Features de composición de matminer para cada fórmula.

    Devuelve la tabla (índice `reduced_formula`, en orden alfabético) y las columnas de cada
    grupo. Los booleanos (p. ej. `compound possible` de IonProperty) se guardan como 0/1.
    """
    makers = _featurizers()
    unknown = set(groups) - set(makers)
    if unknown:
        raise ValueError(f"Grupos de features desconocidos: {sorted(unknown)}")
    formulas = sorted(set(formulas))
    comps = [Composition(f) for f in formulas]
    blocks, columns = [], {}
    for group in groups:
        featurizer = makers[group]()
        with warnings.catch_warnings():
            # matminer avisa de valores imputados y de estados de oxidación poco comunes.
            warnings.simplefilter("ignore")
            rows = [featurizer.featurize(c) for c in comps]
        labels = featurizer.feature_labels()
        blocks.append(pd.DataFrame(rows, columns=labels))
        columns[group] = list(labels)
    table = pd.concat(blocks, axis=1)
    if table.columns.duplicated().any():
        raise ValueError("Nombres de features repetidos entre grupos de matminer.")
    table.index = pd.Index(formulas, name="reduced_formula")
    return table.astype(float), columns


def geometric_features(
    formulas: Iterable[str], shannon: ShannonDict, chi: Mapping[str, float]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """t, μ y τ con el algoritmo de Bartel, y sus metadatos (sitios, oxidación, NC y notas)."""
    formulas = sorted(set(formulas))
    rows = [tau_descriptors(f, shannon, chi).as_row() for f in formulas]
    raw = pd.DataFrame(rows, index=pd.Index(formulas, name="reduced_formula"))
    feats = raw[list(GEOMETRIC_COLUMNS)].rename(columns=GEOMETRIC_COLUMNS).astype(float)
    meta = raw.drop(columns=list(GEOMETRIC_COLUMNS)).add_prefix("bartel_")
    meta["r_A_substitute"] = meta["bartel_A_cn"].notna() & (meta["bartel_A_cn"] != 12)
    meta["r_B_substitute"] = meta["bartel_B_cn"].notna() & (meta["bartel_B_cn"] != 6)
    return feats, meta


def bartel_membership(formulas: Iterable[str], table_s1: pd.DataFrame) -> pd.DataFrame:
    """Pertenencia de cada composición a la TableS1 de Bartel (τ in-sample si está).

    `table_s1` es la salida de `perovskite_pu.data.external.load_bartel2019`. Se compara por
    fórmula reducida. Se excluye toda la TableS1, no solo su partición de entrenamiento: el
    umbral de 4,18 y la forma de τ se eligieron viendo el conjunto completo.
    """
    t = table_s1.drop_duplicates("reduced_formula").set_index("reduced_formula")
    idx = pd.Index(sorted(set(formulas)), name="reduced_formula")
    out = pd.DataFrame(index=idx)
    out["in_bartel2019"] = idx.isin(t.index)
    out["bartel2019_is_train"] = idx.map(t["is_train"]).astype("Float64")
    out["bartel2019_exp_label"] = idx.map(t["exp_label"]).astype("Float64")
    return out


def family_columns(
    families_cfg: Mapping[str, Any], columns_by_group: Mapping[str, list[str]]
) -> dict[str, list[str]]:
    """Columnas de cada familia de features según `features.families` del YAML."""
    out = {}
    for family, spec in families_cfg.items():
        cols = [c for g in spec.get("composition", []) for c in columns_by_group[g]]
        cols += [GEOMETRIC_COLUMNS[g] for g in spec.get("geometric", [])]
        out[family] = cols
    return out


def build_pre_dft(
    formulas: Iterable[str], features_cfg: Mapping[str, Any], table_s1: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Tabla de features, metadatos y columnas por grupo y por familia."""
    formulas = sorted(set(formulas))
    families = features_cfg["families"]
    groups = list(dict.fromkeys(g for spec in families.values() for g in spec["composition"]))
    comp, comp_cols = composition_features(formulas, groups)
    geom, meta = geometric_features(formulas, *default_tables())
    meta = meta.join(bartel_membership(formulas, table_s1))
    table = comp.join(geom)
    columns = {
        "groups": {**comp_cols, "geometric": list(GEOMETRIC_COLUMNS.values())},
        "families": family_columns(families, comp_cols),
    }
    return table, meta, columns


def main(argv: list[str] | None = None) -> None:
    """Calcula las features `pre_dft` de todas las composiciones no polianiónicas."""
    import yaml

    from perovskite_pu.data.external import BARTEL_DIR, load_bartel2019

    cfg = yaml.safe_load((CONFIGS_DIR / "data.yaml").read_text("utf-8"))
    parser = argparse.ArgumentParser(description="F2.1: features pre_dft por composición.")
    parser.add_argument("--labels", default=str(INTERIM_DIR / "f1_5_labels.parquet"))
    args = parser.parse_args(argv)

    labels = pd.read_parquet(args.labels)
    # Todas las composiciones con etiqueta salvo los polianiones (P, U, sintetizadas no
    # perovskita y ambiguas): las dos últimas sirven para la validación de formabilidad.
    formulas = labels.loc[labels["label"] != "polyanion", "reduced_formula"]
    table_s1 = load_bartel2019(BARTEL_DIR / cfg["external_sources"]["bartel2019"]["table"])
    table, meta, columns = build_pre_dft(formulas, cfg["features"], table_s1)

    out = cfg["features"]["output"]
    table.to_parquet(ROOT / out["table"])
    meta.to_parquet(ROOT / out["metadata"])
    (ROOT / out["columns"]).write_text(
        json.dumps(columns, indent=2, ensure_ascii=False) + "\n", "utf-8"
    )
    print(f"{table.shape[0]} composiciones × {table.shape[1]} features -> {out['table']}")
    for family, cols in columns["families"].items():
        print(f"  {family}: {len(cols)} columnas")
    nan_geom = table[list(GEOMETRIC_COLUMNS.values())].isna().sum().to_dict()
    print("NaN por columna geométrica:", nan_geom)


if __name__ == "__main__":  # requisito en Windows (regla 2.2)
    main()
