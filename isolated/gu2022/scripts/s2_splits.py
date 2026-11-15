"""F3.5, S2 (preregistro §10b): partición de Gu et al. agrupando los cristales por composición.

Variante (a): sus 100 particiones originales (`splits_Perov_All.json`), por cristal.
Variante (b): las MISMAS 100 particiones con el MISMO test (PTest y NTest idénticos), pero
quitando de train y validación todo cristal cuya composición reducida aparezca en el test.
Así ΔTPR(a − b) mide solo la fuga entre polimorfos: el test no cambia, solo lo que el modelo
ha visto. La composición sale de `data/Perov_All_ABC.json` (A, B, X de cada cristal; todos son
ABX₃, así que la composición reducida es {A, B, X₃} sin importar qué sitio ocupe cada uno).

Escribe `<work>/splits_S2_grouped.json` (mismo formato que el suyo) y un resumen.

Uso: .venv/bin/python -I scripts/s2_splits.py --repo <repo> --work <work>
"""

# Entorno aislado (no forma parte del paquete): el código reproduce a propósito el de Gu et al.
# (congelación de capas con list(...)[0], semilla global de numpy, open() de sus scripts), así
# que se exime de estas reglas de estilo para no cambiar un comportamiento ya validado.
# ruff: noqa: SIM115

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def composition_keys(repo: Path) -> list[str]:
    """Composición reducida de cada cristal, en el orden de índice de `1PU_Split.py`."""
    with open(repo / "data" / "id_prop.csv") as f:
        rows = sorted(csv.reader(f), key=lambda r: r[0])  # mismo orden que sus índices
    abc = json.load(open(repo / "data" / "Perov_All_ABC.json"))
    keys = []
    for cif_id, _ in rows:
        a, b, x = abc[cif_id]
        keys.append("-".join(sorted((a, b))) + f"|{x}3")
    return keys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    args = ap.parse_args()
    repo, work = args.repo.resolve(), args.work.resolve()
    keys = np.array(composition_keys(repo))
    splits = json.load(open(repo / "splits_Perov_All.json"))
    grouped, removed = [], []
    for s in splits:
        test_comp = set(keys[s["PTest"] + s["NTest"]])
        g = {k: list(v) for k, v in s.items()}
        n_before = sum(len(s[k]) for k in ("PTrain", "NTrain", "PVal", "NVal"))
        for k in ("PTrain", "NTrain", "PVal", "NVal"):
            g[k] = [i for i in s[k] if keys[i] not in test_comp]
        n_after = sum(len(g[k]) for k in ("PTrain", "NTrain", "PVal", "NVal"))
        removed.append({"total": n_before - n_after,
                        "PTrain": len(s["PTrain"]) - len(g["PTrain"]),
                        "NTrain": len(s["NTrain"]) - len(g["NTrain"])})  # fmt: skip
        grouped.append(g)
    json.dump(grouped, open(work / "splits_S2_grouped.json", "w"))
    ptest = splits[0]["PTest"]
    train_comp = set(keys[splits[0]["PTrain"] + splits[0]["PVal"]])
    leak = float(np.mean([keys[i] in train_comp for i in ptest]))
    summary = {
        "n_splits": len(splits),
        "test_identical_in_all": all(s["PTest"] == ptest for s in splits),
        "ptest_with_composition_in_train_val_split0": leak,
        "removed_per_split_mean": {k: float(np.mean([r[k] for r in removed])) for k in removed[0]},
        "removed_per_split_max": {k: int(np.max([r[k] for r in removed])) for k in removed[0]},
        "n_compositions": len(set(keys)),
    }
    (work / "s2_summary.json").write_text(json.dumps(summary, indent=2))
    print("S2", json.dumps(summary))


if __name__ == "__main__":
    main()
