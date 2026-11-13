"""Genera ``tests/fixtures/toy_abo3.csv``: datos SINTÉTICOS para los tests siempre activos.

No son datos reales: las etiquetas y features son aleatorias. Solo sirven para verificar
la mecánica de particiones, fugas y métricas. Para regenerar:

    uv run python tests/fixtures/make_toy_data.py
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

# Elementos de los sitios A y B (combinaciones ficticias; algunas no son químicamente
# razonables, y no importa: el objetivo es la mecánica, no la química).
A_ELEMENTS = ["Ba", "Bi", "Ca", "K", "La", "Na", "Pb", "Sr"]
B_ELEMENTS = ["Al", "Co", "Fe", "Ga", "Mn", "Nb", "Ni", "Ta", "Ti", "Zr"]
SEED = 12345  # semilla fija SOLO para la fixture (no es una semilla de experimento)


def make_toy_frame() -> pd.DataFrame:
    """Crea 80 composiciones ABO3 con etiqueta PU, año y 4 features numéricas."""
    rng = np.random.default_rng(SEED)
    rows = [
        {"composition": f"{a}{b}O3", "A": a, "B": b} for a, b in product(A_ELEMENTS, B_ELEMENTS)
    ]
    df = pd.DataFrame(rows)
    n = len(df)
    df["label"] = (rng.random(n) < 0.35).astype(int)  # ~35 % positivos
    # Año solo para positivos (los no etiquetados no tienen entrada en ICSD).
    df["year"] = np.where(df["label"] == 1, rng.integers(1950, 2025, n), np.nan)
    for j in range(4):
        df[f"f{j}"] = rng.normal(size=n)
    df.loc[rng.choice(n, 5, replace=False), "f3"] = np.nan  # huecos para probar la imputación
    return df


if __name__ == "__main__":
    out = Path(__file__).with_name("toy_abo3.csv")
    # LF explícito para que el archivo sea idéntico en cualquier sistema operativo.
    make_toy_frame().to_csv(out, index=False, lineterminator="\n")
    print(f"Escrito {out}")
