"""F3.5 (exploratorio): rasgos de simetría y distorsión de los 12 907 cristales de Gu et al.

Para cada cristal (identificador de `data/id_prop.csv`) calcula, desde su CIF:

- `spacegroup`: número del grupo espacial (`SpacegroupAnalyzer`, symprec 0,1 Å).
- `crystal_system`: sistema cristalino de ese grupo.
- `is_pm3m`: True si el grupo es Pm-3m (221), el prototipo cúbico ideal.
- `min_bxb`: ángulo B–X–B mínimo (grados) entre dos cationes B que comparten un anión X; los
  enlaces B–X son los que no superan 1,2 veces la distancia B–X más corta del cristal. NaN si
  no hay ningún X compartido por dos B.
- `n_sites`: número de sitios de la celda.

Solo se escriben identificadores y rasgos derivados (nunca estructuras: las de ICSD no se
redistribuyen). Se ejecuta en el entorno aislado, con prioridad baja:

    nice -n 19 .venv/bin/python -I scripts/provenance_features.py --repo <repo> --out <csv>
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np


def bxb_min_angle(structure, b_el: str, x_el: str) -> float:
    """Ángulo B–X–B mínimo (grados) en `structure`; NaN si ningún X une dos B."""
    b_idx = [i for i, s in enumerate(structure) if s.specie.symbol == b_el]
    x_idx = [i for i, s in enumerate(structure) if s.specie.symbol == x_el]
    if not b_idx or not x_idx:
        return float("nan")
    # Distancia B–X más corta del cristal: define el corte de enlace (1,2 veces esa distancia).
    d0 = np.inf
    for i in b_idx:
        for n in structure.get_neighbors(structure[i], 4.0):
            if n.specie.symbol == x_el:
                d0 = min(d0, n.nn_distance)
    if not np.isfinite(d0):
        return float("nan")
    cut = 1.2 * d0
    best = np.inf
    for j in x_idx:
        # Vectores desde el anión X hasta los B enlazados (con imágenes periódicas).
        x_site = structure[j]
        neigh = structure.get_neighbors(x_site, cut)
        vecs = [n.coords - x_site.coords for n in neigh if n.specie.symbol == b_el]
        for u, v in combinations(vecs, 2):
            cosang = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
            best = min(best, float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))))
    return best if np.isfinite(best) else float("nan")


def features(args: tuple[str, str, str, str]) -> dict:
    """Rasgos de un cristal; los errores se registran en la columna `error`."""
    cif_path, cif_id, b_el, x_el = args
    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    row = {"id": cif_id, "spacegroup": "", "crystal_system": "", "is_pm3m": "",
           "min_bxb": "", "n_sites": "", "error": ""}  # fmt: skip
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = Structure.from_file(cif_path)
            sga = SpacegroupAnalyzer(s, symprec=0.1)
            sg = sga.get_space_group_number()
            row.update(spacegroup=sg, crystal_system=sga.get_crystal_system(),
                       is_pm3m=int(sg == 221), n_sites=len(s),
                       min_bxb=round(bxb_min_angle(s, b_el, x_el), 3))  # fmt: skip
    except Exception as exc:  # un CIF problemático no detiene el resto
        row["error"] = type(exc).__name__
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", type=Path, required=True, help="repositorio de Gu descomprimido")
    ap.add_argument("--out", type=Path, required=True, help="CSV de salida")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    repo = args.repo.resolve()
    with open(repo / "data" / "id_prop.csv") as f:
        ids = [r[0] for r in csv.reader(f)]
    with open(repo / "data" / "Perov_All_ABC.json") as f:
        abc = json.load(f)
    tasks = [(str(repo / "data" / f"{i}.cif"), i, abc[i][1], abc[i][2]) for i in ids]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["id", "spacegroup", "crystal_system", "is_pm3m", "min_bxb", "n_sites", "error"]
    with open(args.out, "w", newline="") as f, ProcessPoolExecutor(args.workers) as pool:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for k, row in enumerate(pool.map(features, tasks, chunksize=50), 1):
            w.writerow(row)
            if k % 1000 == 0:
                print(f"{k}/{len(tasks)}", flush=True)
    print("hecho:", args.out)


if __name__ == "__main__":  # requisito en Windows (regla 2.2)
    main()
