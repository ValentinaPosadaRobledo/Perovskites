"""Regenera todas las figuras y tablas del artículo (``uv run poe figures``).

Convención (regla 2.4 de CLAUDE.md):
- Un script por figura o tabla en ``paper/scripts/``, con nombre ``fig_*.py`` o ``tab_*.py``.
- Cada script lee SOLO de ``results/`` (los ``summary.csv``) y escribe en
  ``paper/figures/`` o ``paper/tables/``, junto con un ``.json`` que registra los SHA-256
  de los ``summary.csv`` de entrada y el commit.
- Nada se edita a mano.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent


def main() -> int:
    scripts = sorted(SCRIPTS_DIR.glob("fig_*.py")) + sorted(SCRIPTS_DIR.glob("tab_*.py"))
    if not scripts:
        # Resultado esperado hasta F6.1: todavía no hay figuras.
        print("No hay scripts de figuras o tablas todavía (se crean en F6.1).")
        return 0
    failures = []
    for script in scripts:
        print(f"-> {script.name}")
        # Cada script en su propio proceso: un fallo no contamina a los demás.
        if subprocess.run([sys.executable, str(script)], check=False).returncode != 0:
            failures.append(script.name)
    if failures:
        print(f"Fallaron: {failures}")
        return 1
    print(f"{len(scripts)} figuras/tablas regeneradas.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
