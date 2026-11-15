"""Utilidades comunes de las figuras y tablas del artículo (F6.1).

- Estilo único: paleta categórica de referencia (modo claro, validada para daltonismo en el
  orden fijo azul → naranja → aguamarina; como máximo 3 series por gráfico), trazos finos,
  rejilla y ejes recesivos, texto en tinta neutra (nunca del color de la serie).
- Cada figura sale en inglés (artículo) y en español (tesis): `<nombre>.pdf/.png` y
  `<nombre>_es.pdf/.png`.
- Procedencia: un `.json` por figura o tabla con el SHA-256 de cada CSV de entrada (todos en
  `results/`) y el commit (regla 2.4).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin pantalla: solo archivos
import matplotlib.pyplot as plt

from perovskite_pu.utils.hashing import sha256_file
from perovskite_pu.utils.paths import ROOT

FIG_DIR = ROOT / "paper" / "figures"
TAB_DIR = ROOT / "paper" / "tables"

# Paleta categórica de referencia (modo claro), en su orden fijo: nunca se cicla.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#8a8984", "grid": "#e4e3df"}
SURFACE = "#ffffff"  # fondo blanco para la revista

SCHEMES = ["random", "group_system", "loeo_A", "loeo_B", "family", "cluster", "temporal"]
CONFIRMATORY = ["loeo_A", "loeo_B", "family", "cluster"]
SCHEME_LABEL = {
    "en": {"random": "random", "group_system": "A–B system", "loeo_A": "LOEO-A",
           "loeo_B": "LOEO-B", "family": "B family", "cluster": "cluster",
           "temporal": "temporal"},
    "es": {"random": "aleatorio", "group_system": "sistema A–B", "loeo_A": "LOEO-A",
           "loeo_B": "LOEO-B", "family": "familia de B", "cluster": "clúster",
           "temporal": "temporal"},
}  # fmt: skip


def setup_style() -> None:
    """Parámetros de matplotlib comunes a todas las figuras."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
        "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5, "axes.edgecolor": INK["muted"], "axes.linewidth": 0.6,
        "axes.labelcolor": INK["secondary"], "xtick.color": INK["secondary"],
        "ytick.color": INK["secondary"], "text.color": INK["primary"],
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": INK["grid"], "grid.linewidth": 0.5, "axes.axisbelow": True,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.dpi": 300,
        "legend.frameon": False, "pdf.fonttype": 42,
    })  # fmt: skip


def git_commit() -> str:
    """Commit actual del repositorio."""
    out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                         text=True, check=True)  # fmt: skip
    return out.stdout.strip()


def git_dirty() -> bool:
    """True si hay cambios sin commit en el código o en los resultados de entrada."""
    out = subprocess.run(["git", "status", "--porcelain", "--", "paper/scripts", "results"],
                         cwd=ROOT, capture_output=True, text=True, check=True)  # fmt: skip
    return bool(out.stdout.strip())


def write_provenance(path: Path, inputs: list[Path], note: str = "") -> None:
    """`.json` con los hashes de los CSV de entrada y el commit (regla 2.4)."""
    meta = {
        "commit": git_commit(),
        "dirty": git_dirty(),
        "inputs": {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in inputs},
        "note": note,
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
                                         "utf-8")  # fmt: skip


def save(fig: plt.Figure, name: str, lang: str) -> list[Path]:
    """Guarda la figura en PDF y PNG; sufijo `_es` para la versión en español."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    stem = name if lang == "en" else f"{name}_es"
    paths = [FIG_DIR / f"{stem}.pdf", FIG_DIR / f"{stem}.png"]
    for p in paths:
        # Metadatos fijos en el PDF para que el archivo sea reproducible byte a byte.
        kw = {"metadata": {"CreationDate": None, "ModDate": None}} if p.suffix == ".pdf" else {}
        fig.savefig(p, bbox_inches="tight", **kw)
    plt.close(fig)
    return paths
