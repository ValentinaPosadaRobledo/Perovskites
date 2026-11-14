"""Punto de entrada de F3.4: validación de formabilidad desde su YAML (regla 2.4).

Fija OMP_NUM_THREADS=1 ANTES de importar numpy/sklearn (regla 2.2): la paralelización va solo
en el nivel semilla × variante × fold, con joblib (loky).

Uso:
    uv run python scripts/run_formability.py configs/experiments/f3_4_formability.yaml
    uv run python scripts/run_formability.py <yaml> --dev-mode      # 3 semillas
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # antes de cualquier import numérico

import argparse

from perovskite_pu.evaluation.formability_run import run_formability_experiment


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validación de formabilidad (F3.4).")
    parser.add_argument("config", help="YAML en configs/experiments/")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="permite el árbol de git sucio (guarda el diff)")  # fmt: skip
    parser.add_argument("--dev-mode", action="store_true",
                        help="solo las dev_n_seeds primeras semillas")  # fmt: skip
    args = parser.parse_args(argv)
    run_dir = run_formability_experiment(args.config, allow_dirty=args.allow_dirty,
                                         dev_mode=True if args.dev_mode else None)  # fmt: skip
    print(f"Resultados en {run_dir}")


if __name__ == "__main__":  # requisito en Windows (regla 2.2)
    main()
