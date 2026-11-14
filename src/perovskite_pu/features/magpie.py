"""Features composicionales MAGPIE (matminer `ElementProperty.from_preset("magpie")`).

Se calculan una sola vez por composición y se guardan en `data/interim/magpie.parquet`. Son
features `pre_dft`: solo dependen de la fórmula, no de la etiqueta ni de cálculos DFT.

Las usa F1.7 (clustering del esquema `cluster` y análisis S1) y las reutiliza F2.1. Calcularlas
con todas las composiciones no es fuga: son una transformación fija de la fórmula, sin
parámetros ajustados. El escalado y la PCA sí se ajustan solo con el train de cada fold.

Uso:
    uv run python -m perovskite_pu.features.magpie
"""

from __future__ import annotations

import argparse
import warnings
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from pymatgen.core import Composition

from perovskite_pu.utils.paths import INTERIM_DIR


def magpie_features(formulas: Iterable[str]) -> pd.DataFrame:
    """Tabla de features MAGPIE (una fila por fórmula, indexada por `reduced_formula`)."""
    from matminer.featurizers.composition import ElementProperty  # import tardío (lento)

    featurizer = ElementProperty.from_preset("magpie", impute_nan=True)
    formulas = sorted(set(formulas))
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for f in formulas:
            rows.append(featurizer.featurize(Composition(f)))
    return pd.DataFrame(rows, index=pd.Index(formulas, name="reduced_formula"),
                        columns=featurizer.feature_labels())  # fmt: skip


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Features MAGPIE por composición.")
    parser.add_argument("--labels", type=Path, default=INTERIM_DIR / "f1_5_labels.parquet")
    parser.add_argument("--out", type=Path, default=INTERIM_DIR / "magpie.parquet")
    args = parser.parse_args(argv)
    labels = pd.read_parquet(args.labels)
    formulas = labels.loc[labels["label"] != "polyanion", "reduced_formula"]
    table = magpie_features(formulas)
    table.to_parquet(args.out)
    print(f"{table.shape[0]} composiciones × {table.shape[1]} features MAGPIE -> {args.out}")
    print("valores faltantes:", int(table.isna().sum().sum()))


if __name__ == "__main__":  # requisito en Windows (regla 2.2)
    main()
