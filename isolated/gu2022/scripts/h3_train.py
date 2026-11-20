"""F3.5 (H3 y S3): modelo de grafos de Gu et al. (2022) con bagging PU en nuestros folds.

Protocolo fijado antes de entrenar (preregistro §8, §10b y §12; bitácora del 2026-11-20):

- **Modelo:** la GCNN de Gu (64 features, 1 convolución, grafos de su `CIFData`: radio 7 Å),
  **desde cero** (sin sus pesos base ni capas congeladas), Adam con lr 1e-3, sin decaimiento
  de pesos, lote 512, 50 épocas, y la época de **mejor AUC de validación**, como `2Train.py`.
- **Bagging PU** en cada fold externo: 20 bolsas; cada una con todas las P del train externo
  y K = |P| no etiquetadas tomadas con reemplazo. El 10 % de las P y el 10 % de las U de cada
  bolsa (como su partición 80/10/10) se apartan como validación. Puntuación de test = media
  de las 20 bolsas.
- **Sin bucle interno:** H3 compara el grafo con el PU composicional a igual P(ŷ=1) en cada
  fold (§7.2), así que solo hacen falta las puntuaciones de test.
- **Semillas:** de cada semilla del proyecto, `SeedSequence(seed).spawn(3)` → (folds,
  muestreo PU, inicialización); cada bolsa extiende la `spawn_key` con (esquema, fold, bolsa),
  como `perovskite_pu.training.runner.task_seed`. Iguales para A y S3 (números aleatorios
  comunes).

Entradas (paquete de `scripts/h3_structures.py`): `<paquete>/{A,S3}/` (CIF e `id_prop.csv`),
`folds.csv` y `seeds.json`. Salida, reanudable: `<salida>/<criterio>/parts/
seed_<s>_<esquema>_<fold>.csv` con una fila por composición de test (formato `raw/` del
proyecto: baseline, scheme, seed, fold, composition, label, score), escrita de forma atómica.

    .venv/bin/python -I scripts/h3_train.py --repo <repo de Gu> --shim <shim> \
        --package <paquete> --out <salida> --criterion A [--seeds 0 1] [--schemes random] \
        [--max-folds N] [--device auto] [--threads 4]
"""

# Entorno aislado (no forma parte del paquete del proyecto): reutiliza el código de Gu tal
# cual (semilla global de numpy, cargadores con SubsetRandomSampler).
# ruff: noqa: NPY002

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from time import time

import numpy as np

SCHEMES = ("random", "group_system", "loeo_A", "loeo_B", "family", "cluster", "temporal")
STREAMS = ("fold_assignment", "pu_sampling", "model_init")  # orden de utils/seeds.py
N_BAGS, VAL_FRACTION, EPOCHS, BATCH, LR = 20, 0.10, 50, 512, 1e-3


def task_seq(seed: int, stream: str, scheme: str, fold: int, bag: int) -> np.random.SeedSequence:
    """SeedSequence de (semilla, flujo, esquema, fold, bolsa), como `runner.task_seed`."""
    parent = np.random.SeedSequence(seed).spawn(len(STREAMS))[STREAMS.index(stream)]
    key = (*parent.spawn_key, SCHEMES.index(scheme), int(fold), int(bag))
    return np.random.SeedSequence(entropy=parent.entropy, spawn_key=key)


def read_folds(path: Path) -> dict[tuple[int, str, int], list[str]]:
    """Composiciones de test de cada (semilla, esquema, fold)."""
    out: dict[tuple[int, str, int], list[str]] = defaultdict(list)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out[(int(r["seed"]), r["scheme"], int(r["fold"]))].append(r["composition"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("repo", "shim", "package", "out"):
        ap.add_argument(f"--{name}", type=Path, required=True)
    ap.add_argument("--criterion", choices=("A", "S3"), required=True)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--schemes", nargs="*", default=None)
    ap.add_argument("--max-folds", type=int, default=None, help="solo para medir tiempos")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    repo, pkg = args.repo.resolve(), args.package.resolve()
    sys.path.insert(0, str(args.shim.resolve()))
    sys.path.insert(0, str(repo))
    import torch
    from gcnn.data import CIFData, Parallel_Collate_Pool, get_loader
    from gcnn.model import GCNN
    from sklearn import metrics

    torch.set_num_threads(args.threads)
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    spec = importlib.util.spec_from_file_location("gu_train", repo / "2Train.py")
    gu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gu)  # solo define funciones (use_model)

    # Grafos de todas las estructuras del criterio, con caché propia junto a la salida.
    struct_dir = pkg / args.criterion
    out_dir = args.out.resolve() / args.criterion
    (out_dir / "parts").mkdir(parents=True, exist_ok=True)
    data = CIFData(str(struct_dir), cache_path=str(out_dir / "cache"))
    pos_of = {d[2]: k for k, d in enumerate(data.data)}
    with open(struct_dir / "id_prop.csv", newline="") as f:
        label = {r[0]: int(r[1]) for r in csv.reader(f)}
    seeds = (
        args.seeds
        if args.seeds is not None
        else json.loads((pkg / "seeds.json").read_text())["seeds"]
    )
    folds = read_folds(pkg / "folds.csv")
    collate = Parallel_Collate_Pool(1, data.orig_atom_fea_len, data.nbr_fea_len)
    available = set(label)

    tasks = sorted(
        k for k in folds if k[0] in seeds and (args.schemes is None or k[1] in args.schemes)
    )
    if args.max_folds is not None:
        tasks = tasks[: args.max_folds]
    log = {"criterion": args.criterion, "device": device, "n_tasks": len(tasks), "bags": []}
    for seed, scheme, fold in tasks:
        part = out_dir / "parts" / f"seed_{seed}_{scheme}_{fold:03d}.csv"
        if part.exists():  # reanudable
            continue
        # Test: las composiciones del fold con estructura; train: el resto con estructura.
        test = sorted(set(folds[(seed, scheme, fold)]) & available)
        all_fold = {
            c for (s, sch, _), cs in folds.items() if s == seed and sch == scheme for c in cs
        }
        train = sorted((all_fold & available) - set(test))
        if not test:
            continue
        p_train = [c for c in train if label[c] == 1]
        u_train = [c for c in train if label[c] == 0]
        scores = np.zeros(len(test))
        t_fold = time()
        for bag in range(N_BAGS):
            rng = np.random.default_rng(task_seq(seed, "pu_sampling", scheme, fold, bag))
            u_bag = list(rng.choice(u_train, size=len(p_train), replace=True))
            p_bag = list(rng.permutation(p_train))
            n_pv, n_uv = round(VAL_FRACTION * len(p_bag)), round(VAL_FRACTION * len(u_bag))
            val = p_bag[:n_pv] + u_bag[:n_uv]
            trn = p_bag[n_pv:] + u_bag[n_uv:]
            # Inicialización y orden de los lotes: flujo `model_init` de esta bolsa.
            s_py, s_np, s_torch = (
                int(x) for x in task_seq(seed, "model_init", scheme, fold, bag).generate_state(3)
            )
            random.seed(s_py)
            np.random.seed(s_np % 2**32)
            torch.manual_seed(s_torch)
            idx = lambda cs: [pos_of[c] for c in cs]  # noqa: E731
            train_l, val_l, test_l = get_loader(
                data, collate, BATCH, [idx(trn), idx(val), idx(test)], 0, True
            )
            model = GCNN(data.orig_atom_fea_len, data.nbr_fea_len, 64, 1).to(device)  # desde cero
            crit = torch.nn.BCELoss()
            opt = torch.optim.Adam(model.parameters(), LR, weight_decay=0)
            best, best_state, t0 = -1.0, None, time()
            for ep in range(EPOCHS):
                gu.use_model(train_l, model, crit, opt, ep, "train")
                out, tgt, _ = gu.use_model(val_l, model, crit, opt, ep, "predict", "Val")
                auc = metrics.roc_auc_score(tgt, out) if len(set(tgt)) == 2 else float("nan")
                if auc > best or best_state is None:
                    best = auc
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(best_state)
            out, _, ids = gu.use_model(test_l, model, crit, opt, 0, "predict", "Test")
            by_id = dict(
                zip(
                    ids,
                    (float(np.ravel(o)[0]) if np.ndim(o) else float(o) for o in out),
                    strict=True,
                )
            )
            scores += np.array([by_id[c] for c in test])
            log["bags"].append(
                {
                    "seed": seed,
                    "scheme": scheme,
                    "fold": fold,
                    "bag": bag,
                    "n_train": len(trn),
                    "best_val_auc": best,
                    "train_s": round(time() - t0, 2),
                }
            )
        scores /= N_BAGS
        tmp = part.with_suffix(".tmp")
        with open(tmp, "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["baseline", "scheme", "seed", "fold", "composition", "label", "score"])
            for c, s in zip(test, scores, strict=True):
                w.writerow([f"graph_{args.criterion}", scheme, seed, fold, c, label[c], f"{s:.8f}"])
        os.replace(tmp, part)  # escritura atómica: una parte a medias nunca queda como terminada
        print(
            f"{part.name}: {len(test)} test, {len(p_train)} P de train, {time() - t_fold:.1f} s",
            flush=True,
        )
    stamp = int(time())
    (out_dir / f"log_{stamp}.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":  # requisito en Windows (regla 2.2)
    main()
