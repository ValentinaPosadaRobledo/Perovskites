"""F3.5: fidelidad de nuestro entorno frente a las predicciones guardadas por Gu et al.

Con SUS pesos entrenados de la partición `--split`, predice las positivas de test (PTest) y
las compara, alineadas por identificador de cristal, con su `tests/Y_Perov_All_xxx.json`
(una puntuación por positiva de test, en el orden de PTest). Reutiliza la caché de grafos que
construyó `time_one_model.py` (nuestra, no la suya).

Uso: .venv/bin/python -I scripts/check_fidelity.py --repo <repo> --work <work> --shim <shim>
"""

# Entorno aislado (no forma parte del paquete): el código reproduce a propósito el de Gu et al.
# (congelación de capas con list(...)[0], semilla global de numpy, open() de sus scripts), así
# que se exime de estas reglas de estilo para no cambiar un comportamiento ya validado.
# ruff: noqa: SIM115

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--shim", type=Path, required=True)
    ap.add_argument("--split", type=int, default=0)
    args = ap.parse_args()
    repo, work = args.repo.resolve(), args.work.resolve()
    sys.path.insert(0, str(args.shim.resolve()))
    sys.path.insert(0, str(repo))
    import torch
    from gcnn.data import CIFData, Parallel_Collate_Pool, get_loader
    from gcnn.model import GCNN

    splits = json.load(open(repo / "splits_Perov_All.json"))[args.split]
    with open(repo / "data" / "id_prop.csv") as f:
        rows = sorted(csv.reader(f), key=lambda r: r[0])
    data = CIFData(str(work / f"subset_{args.split:03d}"),
                   cache_path=str(work / f"cache_{args.split:03d}"))  # fmt: skip
    pos_of = {d[2]: k for k, d in enumerate(data.data)}
    ptest_ids = [rows[i][0] for i in splits["PTest"]]
    collate_fn = Parallel_Collate_Pool(1, data.orig_atom_fea_len, data.nbr_fea_len)
    (loader,) = get_loader(data, collate_fn, 512, [[pos_of[c] for c in ptest_ids]], 0, True)

    model = GCNN(data.orig_atom_fea_len, data.nbr_fea_len, 64, 1)
    state = torch.load(repo / "weights" / f"W_Perov_All_{args.split:03d}.pth.tar",
                       map_location="cpu", weights_only=True)  # fmt: skip
    model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()})
    model.eval()
    ours = {}
    with torch.no_grad():
        for inputs, _target, ids, _ in loader:
            out, _ = model(*inputs)
            ours.update(zip(ids, out.tolist(), strict=True))
    theirs = json.load(open(repo / "tests" / f"Y_Perov_All_{args.split:03d}.json"))
    a = np.array([ours[c] for c in ptest_ids])
    b = np.array(theirs, dtype=float)
    rep = {
        "n": int(a.size),
        "max_abs_diff": float(np.abs(a - b).max()),
        "mean_abs_diff": float(np.abs(a - b).mean()),
        "pearson": float(np.corrcoef(a, b)[0, 1]),
        "tpr_at_0.5_ours": float((a > 0.5).mean()),
        "tpr_at_0.5_theirs": float((b > 0.5).mean()),
        "agree_at_0.5": float(((a > 0.5) == (b > 0.5)).mean()),
    }
    sa, sb = np.sort(a), np.sort(b)
    rep["sorted_max_abs_diff"] = float(np.abs(sa - sb).max())
    rep["sorted_mean_abs_diff"] = float(np.abs(sa - sb).mean())
    print("FIDELIDAD", json.dumps(rep))
    (work / f"fidelity_{args.split:03d}.json").write_text(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
