"""Punto de entrada para ejecutar un experimento desde su YAML (regla 2.4).

Fija OMP_NUM_THREADS=1 ANTES de importar numpy/sklearn (regla 2.2): la paralelización va solo
en el nivel semilla × esquema, con joblib (loky).

Uso:
    uv run python scripts/run_experiment.py configs/experiments/f2_4_baselines.yaml
    uv run python scripts/run_experiment.py <yaml> --allow-dirty   # guarda el diff
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # antes de cualquier import numérico

import argparse

from perovskite_pu.training.runner import run_experiment


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ejecuta un experimento desde su YAML.")
    parser.add_argument("config", help="YAML en configs/experiments/")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="permite el árbol de git sucio (guarda el diff con los resultados)",
    )
    args = parser.parse_args(argv)
    run_dir = run_experiment(args.config, allow_dirty=args.allow_dirty)
    print(f"Resultados en {run_dir}")


if __name__ == "__main__":  # requisito en Windows (regla 2.2)
    main()
