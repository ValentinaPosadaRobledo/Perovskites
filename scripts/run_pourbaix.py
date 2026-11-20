"""Punto de entrada de F5.1: ΔG_pbx de las perovskitas de train_pool desde su YAML (regla 2.4).

Fija OMP_NUM_THREADS=1 ANTES de importar numpy (regla 2.2): la paralelización va solo en el
nivel de composición, con joblib (loky). Requiere la descarga previa:
`uv run python -m perovskite_pu.pourbaix.download`.

Uso:
    uv run python scripts/run_pourbaix.py configs/experiments/f5_1_pourbaix.yaml
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # antes de cualquier import numérico

import argparse

from perovskite_pu.pourbaix.run import run_pourbaix_experiment


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="ΔG_pbx de las perovskitas (F5.1).")
    parser.add_argument("config", help="YAML en configs/experiments/")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="permite el árbol de git sucio (guarda el diff)")  # fmt: skip
    args = parser.parse_args(argv)
    run_dir = run_pourbaix_experiment(args.config, allow_dirty=args.allow_dirty)
    print(f"Resultados en {run_dir}")


if __name__ == "__main__":  # requisito en Windows (regla 2.2)
    main()
