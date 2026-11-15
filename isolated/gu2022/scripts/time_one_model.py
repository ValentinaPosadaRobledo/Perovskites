"""F3.5, paso de medición: coste en CPU del modelo de Gu et al. (2022) en el entorno aislado.

1. Comprueba el módulo de compatibilidad de `torch_scatter` frente a bucles ingenuos.
2. Construye los grafos (VoronoiNN, código de Gu) de los cristales de la partición 0
   (train + val + test) desde los CIF, NO desde su `data_cache.pkl`, y mide el tiempo.
3. Fidelidad: con SUS pesos ya entrenados (`weights/W_Perov_All_000.pth.tar`, cargados con
   `weights_only=True`), predice el test y lo compara con su `tests/Y_Perov_All_000.json`.
4. Entrena un modelo como `2Train.py` (transferencia desde `base_weights/W0_000`, 50 épocas,
   CPU, semillas fijas) y mide el tiempo por época y el total.

Uso (desde el entorno aislado, nunca desde el repositorio de la tesis):
    .venv/bin/python -I scripts/time_one_model.py --repo <repo de Gu> --work <dir de trabajo>
"""

# Entorno aislado (no forma parte del paquete): el código reproduce a propósito el de Gu et al.
# (congelación de capas con list(...)[0], semilla global de numpy, open() de sus scripts), así
# que se exime de estas reglas de estilo para no cambiar un comportamiento ya validado.
# ruff: noqa: SIM115, NPY002, RUF015, RUF059, E501

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import random
import sys
from pathlib import Path
from time import time

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--shim", type=Path, required=True)
    ap.add_argument("--split", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=50)
    args = ap.parse_args()
    repo, work = args.repo.resolve(), args.work.resolve()

    # Se importa a propósito el código de Gu (paso aprobado por la autora) y el módulo de
    # compatibilidad de torch_scatter.
    sys.path.insert(0, str(args.shim.resolve()))
    sys.path.insert(0, str(repo))
    import torch
    from sklearn import metrics

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    report: dict = {"torch": torch.__version__, "threads": torch.get_num_threads()}

    # 1. Módulo de compatibilidad frente a bucles ingenuos.
    from torch_scatter import scatter_mean, scatter_min

    src = torch.randn(50, 3)
    idx = torch.randint(0, 7, (50,))
    m = scatter_mean(src, idx, dim=0, dim_size=9)
    for g in range(9):
        sel = src[idx == g]
        ref = sel.mean(0) if len(sel) else torch.zeros(3)
        assert torch.allclose(m[g], ref, atol=1e-6)
    v1 = torch.randn(40)
    i1 = torch.randint(0, 5, (40,))
    mn, arg = scatter_min(v1, i1, dim_size=6)
    for g in range(6):
        sel = v1[i1 == g]
        if len(sel):
            assert torch.isclose(mn[g], sel.min()) and v1[arg[g]] == sel.min()
        else:
            assert mn[g] == 0 and arg[g] == 40
    report["shim_check"] = "ok"

    # 2. Grafos de la partición elegida, desde los CIF.
    splits = json.load(open(repo / "splits_Perov_All.json"))[args.split]
    with open(repo / "data" / "id_prop.csv") as f:
        rows = sorted(csv.reader(f), key=lambda r: r[0])  # mismo orden que 1PU_Split.py
    parts = {k: splits[k] for k in ("PTrain", "NTrain", "PVal", "NVal", "PTest", "NTest")}
    needed = sorted({i for v in parts.values() for i in v})
    sub = work / f"subset_{args.split:03d}"
    sub.mkdir(parents=True, exist_ok=True)
    with open(sub / "id_prop.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows[i] for i in needed)
    for i in needed:
        link = sub / f"{rows[i][0]}.cif"
        if not link.exists():
            os.symlink(repo / "data" / f"{rows[i][0]}.cif", link)
    from gcnn.data import CIFData, Parallel_Collate_Pool, get_loader
    from gcnn.model import GCNN

    t0 = time()
    data = CIFData(str(sub), cache_path=str(work / f"cache_{args.split:03d}"))
    report["featurize_s"] = time() - t0
    report["n_crystals"] = len(data)
    report["featurize_s_per_1000"] = report["featurize_s"] / len(data) * 1000
    pos_of = {rows[i][0]: k for k, i in enumerate(needed)}  # id → posición en el subconjunto
    assert [d[2] for d in data.data] == [rows[i][0] for i in needed], "orden distinto"
    remap = {k: [pos_of[rows[i][0]] for i in v] for k, v in parts.items()}

    collate_fn = Parallel_Collate_Pool(1, data.orig_atom_fea_len, data.nbr_fea_len)
    train_l, val_l, tpos_l, tneg_l = get_loader(
        data, collate_fn, 512,
        [remap["PTrain"] + remap["NTrain"], remap["PVal"] + remap["NVal"],
         remap["PTest"], remap["NTest"]], 0, True,
    )  # fmt: skip
    spec = importlib.util.spec_from_file_location("gu_train", repo / "2Train.py")
    gu_train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gu_train)  # solo define funciones (main está bajo __name__)
    criterion = torch.nn.BCELoss()

    def load_gu_state(path: Path) -> dict:
        """Pesos de Gu (guardados con nn.DataParallel): se quita el prefijo `module.`."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        return {k.removeprefix("module."): v for k, v in state.items()}

    def new_model() -> torch.nn.Module:
        """GCNN con la configuración de 2Train.py (64 features, 1 convolución)."""
        return GCNN(data.orig_atom_fea_len, data.nbr_fea_len, 64, 1)

    # 3. Fidelidad con sus pesos entrenados.
    model = new_model()
    model.load_state_dict(load_gu_state(repo / "weights" / f"W_Perov_All_{args.split:03d}.pth.tar"))
    out_p, _, ids_p = gu_train.use_model(tpos_l, model, criterion, None, 0, "predict", "TestP")
    out_n, _, ids_n = gu_train.use_model(tneg_l, model, criterion, None, 0, "predict", "TestN")
    theirs = json.load(open(repo / "tests" / f"Y_Perov_All_{args.split:03d}.json"))
    report["their_test_json_len"] = len(theirs)
    report["our_test_scores_head"] = [round(x, 4) for x in out_p[:5]]
    report["their_test_json_head"] = theirs[:5] if isinstance(theirs, list) else str(theirs)[:200]
    y = [1] * len(out_p) + [0] * len(out_n)
    report["their_weights_test_auc"] = metrics.roc_auc_score(y, out_p + out_n)
    report["their_weights_test_tpr_at_0.5"] = float(np.mean(np.array(out_p) > 0.5))

    # 4. Entrenamiento de un modelo (transferencia como en 2Train.py), en CPU.
    model = new_model()
    for layer in list(list(list(list(model.children())[0].children())[0].children())[0].children())[1:9]:  # fmt: skip
        for p in layer.parameters():
            p.requires_grad = False
    model.load_state_dict(load_gu_state(repo / "base_weights" / f"W0_{args.split:03d}.pth.tar"))
    opt = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=0)
    epoch_s, best = [], 0.0
    for ep in range(args.epochs):
        t1 = time()
        gu_train.use_model(train_l, model, criterion, opt, ep, "train")
        out, tgt, _ = gu_train.use_model(val_l, model, criterion, opt, ep, "predict", "Val")
        best = max(best, metrics.roc_auc_score(tgt, out))
        epoch_s.append(time() - t1)
    report["epoch_s_mean"] = float(np.mean(epoch_s))
    report["train_total_s"] = float(np.sum(epoch_s))
    report["best_val_auc"] = best
    (work / f"timing_{args.split:03d}.json").write_text(json.dumps(report, indent=2))
    print("RESULTADO", json.dumps(report))


if __name__ == "__main__":
    main()
