"""Punto de entrada de F5.2: toxicidad y frente de Pareto desde su YAML (regla 2.4).

Fija OMP_NUM_THREADS=1 ANTES de importar numpy (regla 2.2). Requiere las corridas de F5.1 y
F3.3 registradas en results/runs_index.csv.

Uso:
    uv run python scripts/run_pareto.py configs/experiments/f5_2_pareto.yaml
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # antes de cualquier import numérico

import argparse

from perovskite_pu.pourbaix.pareto_run import run_pareto_experiment


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Toxicidad y frente de Pareto (F5.2).")
    parser.add_argument("config", help="YAML en configs/experiments/")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="permite el árbol de git sucio (guarda el diff)")  # fmt: skip
    args = parser.parse_args(argv)
    run_dir = run_pareto_experiment(args.config, allow_dirty=args.allow_dirty)
    print(f"Resultados en {run_dir}")


if __name__ == "__main__":  # requisito en Windows (regla 2.2)
    main()
