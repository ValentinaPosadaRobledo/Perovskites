"""F3.5, paso 1: reproducir el ensemble de 100 modelos de Gu et al. (2022) solo con inferencia.

1. Construye los grafos de los 12 907 cristales desde los CIF (código de Gu; caché propia).
2. Con SUS 100 pesos entrenados (`weights/W_Perov_All_xxx.pth.tar`, `weights_only=True`),
   puntúa todos los cristales; media y DE por cristal = su "CL score".
3. Compara con su `predict/Perov_All.json` ([ids, media, etiqueta, DE]) por identificador.
4. TPR del ensemble en su test (PTest, el mismo en las 100 particiones) alineando por
   cristal, frente a la cifra que da su README, que promedia `tests/*.json` posición a posición
   (cada archivo está en el orden aleatorio de su cargador).

Uso: .venv/bin/python -I scripts/reproduce_ensemble.py --repo <repo> --work <work> --shim <shim>
"""

# Entorno aislado (no forma parte del paquete): el código reproduce a propósito el de Gu et al.
# (congelación de capas con list(...)[0], semilla global de numpy, open() de sus scripts), así
# que se exime de estas reglas de estilo para no cambiar un comportamiento ya validado.
# ruff: noqa: SIM115, RUF059, E501

from __future__ import annotations

import argparse
import csv
import json
import sys
from glob import glob
from pathlib import Path
from time import time

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--shim", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    repo, work = args.repo.resolve(), args.work.resolve()
    sys.path.insert(0, str(args.shim.resolve()))
    sys.path.insert(0, str(repo))
    import torch
    from gcnn.data import CIFData, Parallel_Collate_Pool, get_loader
    from gcnn.model import GCNN

    torch.set_num_threads(args.threads)
    rep: dict = {}
    t0 = time()
    data = CIFData(str(repo / "data"), cache_path=str(work / "cache_all"))
    rep["featurize_or_load_s"] = time() - t0
    ids = [d[2] for d in data.data]
    collate_fn = Parallel_Collate_Pool(1, data.orig_atom_fea_len, data.nbr_fea_len)
    (loader,) = get_loader(data, collate_fn, 512, [list(range(len(data)))], 0, True)
    batches = list(loader)  # mismos lotes para los 100 modelos

    scores = np.zeros((100, len(ids)))
    pos_of = {c: k for k, c in enumerate(ids)}
    t1 = time()
    for m in range(100):
        model = GCNN(data.orig_atom_fea_len, data.nbr_fea_len, 64, 1)
        st = torch.load(repo / "weights" / f"W_Perov_All_{m:03d}.pth.tar", map_location="cpu",
                        weights_only=True)  # fmt: skip
        model.load_state_dict({k.removeprefix("module."): v for k, v in st.items()})
        model.eval()
        with torch.no_grad():
            for inputs, _t, bid, _ in batches:
                out, _ = model(*inputs)
                scores[m, [pos_of[c] for c in bid]] = out.numpy()
    rep["inference_s"] = time() - t1
    mean, std = scores.mean(0), scores.std(0)
    np.save(work / "ensemble_scores.npy", scores)

    # 3. Frente a su predict/Perov_All.json.
    p_ids, p_mean, p_target, p_std = json.load(open(repo / "predict" / "Perov_All.json"))
    ours = dict(zip(ids, mean, strict=True))
    ours_sd = dict(zip(ids, std, strict=True))
    a = np.array([ours[c] for c in p_ids])
    rep["n_compared"] = len(p_ids)
    rep["cl_max_abs_diff"] = float(np.abs(a - np.array(p_mean)).max())
    rep["std_max_abs_diff"] = float(np.abs(np.array([ours_sd[c] for c in p_ids]) - np.array(p_std)).max())  # fmt: skip

    # 4. TPR del ensemble en su test, alineado por cristal frente a su cálculo del README.
    split0 = json.load(open(repo / "splits_Perov_All.json"))[0]
    with open(repo / "data" / "id_prop.csv") as f:
        rows = sorted(csv.reader(f), key=lambda r: r[0])
    ptest = [rows[i][0] for i in split0["PTest"]]
    ntest = [rows[i][0] for i in split0["NTest"]]
    same_test = all(
        [rows[i][0] for i in s["PTest"]] == ptest
        for s in json.load(open(repo / "splits_Perov_All.json"))
    )
    rep["ptest_same_in_all_splits"] = same_test
    e = np.array([ours[c] for c in ptest])
    rep["ensemble_tpr_aligned"] = float((e > 0.5).mean())
    rep["ensemble_unlabeled_test_ppr_aligned"] = float(
        (np.array([ours[c] for c in ntest]) > 0.5).mean()
    )
    theirs = [json.load(open(p)) for p in sorted(glob(str(repo / "tests" / "*.json")))]
    rep["readme_ensemble_tpr_positionwise"] = float((np.mean(theirs, axis=0) > 0.5).mean())
    rep["mean_single_model_tpr_theirs"] = float(np.mean([(np.array(t) > 0.5).mean() for t in theirs]))  # fmt: skip
    print("ENSEMBLE", json.dumps(rep))
    (work / "ensemble_report.json").write_text(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
