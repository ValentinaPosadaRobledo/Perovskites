"""F3.5: entrena UN modelo de Gu et al. (2022) en una partición, como su `2Train.py`, en CPU.

Mismo modelo (GCNN, 64 features, 1 convolución), transferencia desde
`base_weights/W0_<split>.pth.tar` con las mismas capas congeladas, Adam (lr 1e-3), 50 épocas,
lote 512 y el mejor AUC de validación. Diferencias declaradas: CPU, semillas fijas
(`--seed`, que el orquestador lee de `configs/seeds.yaml` del proyecto) y que las predicciones
de test se guardan **alineadas por identificador de cristal** (las suyas quedaban en el orden
aleatorio del cargador).

Uso: .venv/bin/python -I scripts/train_split.py --repo <repo> --work <work> --shim <shim>
         --splits <archivo json> --split <i> --variant <a|b> --seed <semilla>
"""

# Entorno aislado (no forma parte del paquete): el código reproduce a propósito el de Gu et al.
# (congelación de capas con list(...)[0], semilla global de numpy, open() de sus scripts), así
# que se exime de estas reglas de estilo para no cambiar un comportamiento ya validado.
# ruff: noqa: SIM115, NPY002, RUF015, E501

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import sys
from pathlib import Path
from time import time

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    for name in ("repo", "work", "shim", "splits"):
        ap.add_argument(f"--{name}", type=Path, required=True)
    ap.add_argument("--split", type=int, required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    repo, work = args.repo.resolve(), args.work.resolve()
    out_dir = work / f"S2_{args.variant}"
    out_dir.mkdir(parents=True, exist_ok=True)
    done = out_dir / f"test_{args.split:03d}.json"
    if done.exists():  # reanudable: un modelo terminado no se repite
        return
    sys.path.insert(0, str(args.shim.resolve()))
    sys.path.insert(0, str(repo))
    import torch
    from sklearn import metrics

    torch.set_num_threads(args.threads)
    # Semillas de esta tarea: derivadas de (semilla del proyecto, partición) sin colisiones.
    ss = np.random.SeedSequence([args.seed, args.split])
    s_py, s_np, s_torch = (int(x) for x in ss.generate_state(3))
    random.seed(s_py)
    np.random.seed(s_np % 2**32)
    torch.manual_seed(s_torch)

    from gcnn.data import CIFData, Parallel_Collate_Pool, get_loader
    from gcnn.model import GCNN

    data = CIFData(str(repo / "data"), cache_path=str(work / "cache_all"))  # caché propia
    pos_of = {d[2]: k for k, d in enumerate(data.data)}
    with open(repo / "data" / "id_prop.csv") as f:
        rows = sorted(csv.reader(f), key=lambda r: r[0])  # orden de índices de 1PU_Split.py
    sp = json.load(open(args.splits))[args.split]

    def idx(keys: tuple[str, ...]) -> list[int]:
        """Índices de `CIFData` de las listas de la partición."""
        return [pos_of[rows[i][0]] for k in keys for i in sp[k]]

    collate_fn = Parallel_Collate_Pool(1, data.orig_atom_fea_len, data.nbr_fea_len)
    train_l, val_l, test_l = get_loader(
        data, collate_fn, 512,
        [idx(("PTrain", "NTrain")), idx(("PVal", "NVal")), idx(("PTest", "NTest"))], 0, True,
    )  # fmt: skip
    spec = importlib.util.spec_from_file_location("gu_train", repo / "2Train.py")
    gu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gu)  # solo define funciones

    model = GCNN(data.orig_atom_fea_len, data.nbr_fea_len, 64, 1)
    # Transferencia como en 2Train.py: se congelan el embedding y la primera capa.
    for layer in list(list(list(list(model.children())[0].children())[0].children())[0].children())[1:9]:  # fmt: skip
        for p in layer.parameters():
            p.requires_grad = False
    st = torch.load(repo / "base_weights" / f"W0_{args.split:03d}.pth.tar", map_location="cpu",
                    weights_only=True)  # fmt: skip
    model.load_state_dict({k.removeprefix("module."): v for k, v in st.items()})
    criterion = torch.nn.BCELoss()
    opt = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=0)
    best, best_state, t0 = -1.0, None, time()
    for ep in range(args.epochs):
        gu.use_model(train_l, model, criterion, opt, ep, "train")
        out, tgt, _ = gu.use_model(val_l, model, criterion, opt, ep, "predict", "Val")
        auc = metrics.roc_auc_score(tgt, out)
        if auc > best:  # como 2Train.py: se guarda el mejor por AUC de validación
            best = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    out, tgt, ids = gu.use_model(test_l, model, criterion, opt, 0, "predict", "Test")
    rec = {"split": args.split, "variant": args.variant, "seed": args.seed,
           "best_val_auc": best, "train_s": time() - t0,
           "n_train": len(train_l.sampler), "scores": dict(zip(ids, out, strict=True)),
           "labels": dict(zip(ids, tgt, strict=True))}  # fmt: skip
    done.write_text(json.dumps(rec))


if __name__ == "__main__":
    main()
