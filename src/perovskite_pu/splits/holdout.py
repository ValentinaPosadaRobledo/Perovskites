"""F1.8: holdout externo congelado, definido por grupos de elementos B (regla 2.1.5).

Selección (determinista con `holdout_seed`):
1. Candidatos: elementos B con al menos una positiva en el pool, agrupados por familia.
2. En cada familia, los candidatos se ordenan al azar con un generador derivado de
   `holdout_seed` y se añaden hasta cubrir `target` × (positivas de la familia).
3. Si la fracción total de positivas reservadas queda fuera de [mín, máx], se repite con el
   siguiente orden aleatorio del mismo generador (determinista). Se registra el intento.

El holdout son TODAS las composiciones del pool PU cuyo B está en el conjunto reservado.
`train_pool` es el resto. Ninguna composición puede estar en ambos.

Este módulo crea el holdout; solo `perovskite_pu.evaluation.holdout.load_holdout` lo LEE.

`rebuild` regenera `train_pool` en un contenedor nuevo (data/processed/ no se versiona) sin
volver a congelar: usa los B reservados de `HASHES.json` y verifica en memoria los dos hashes
congelados antes de escribir nada.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class HoldoutSelection:
    """Elementos B reservados y cifras de la selección."""

    reserved_b: list[str]
    attempt: int
    positive_fraction: float
    per_family: dict[str, list[str]] = field(default_factory=dict)


def select_reserved_b(
    pool: pd.DataFrame,
    holdout_seed: int,
    target: float = 0.125,
    bounds: Sequence[float] = (0.10, 0.15),
    max_attempts: int = 1000,
) -> HoldoutSelection:
    """Elige los elementos B reservados, estratificando por familia (ver docstring del módulo)."""
    pos = pool[pool["label"] == 1]
    total = len(pos)
    pos_by_b = pos.groupby("B").size()
    fam_of_b = pool.drop_duplicates("B").set_index("B")["family"]
    rng = np.random.default_rng(np.random.SeedSequence(holdout_seed))
    for attempt in range(1, max_attempts + 1):
        reserved, per_family = [], {}
        for fam in sorted(fam_of_b.unique()):
            cands = sorted(b for b in pos_by_b.index if fam_of_b.get(b) == fam)
            goal = target * pos_by_b[cands].sum() if cands else 0
            chosen, acc = [], 0
            for b in rng.permutation(cands):
                if acc >= goal:
                    break
                chosen.append(str(b))
                acc += int(pos_by_b[b])
            per_family[fam] = sorted(chosen)
            reserved += chosen
        frac = float(pos_by_b[reserved].sum() / total) if reserved else 0.0
        if bounds[0] <= frac <= bounds[1]:
            return HoldoutSelection(sorted(reserved), attempt, frac, per_family)
    raise RuntimeError(f"Sin selección en [{bounds[0]}, {bounds[1]}] tras {max_attempts} intentos.")


def split_holdout(
    pool: pd.DataFrame, reserved_b: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(holdout, train_pool): el holdout son todas las composiciones con B reservado."""
    mask = pool["B"].isin(set(reserved_b))
    holdout, train_pool = pool[mask].copy(), pool[~mask].copy()
    if set(holdout["composition"]) & set(train_pool["composition"]):
        raise AssertionError("Composiciones en holdout y train_pool a la vez.")
    return holdout.reset_index(drop=True), train_pool.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Congelación (una sola vez) y folds definitivos sobre train_pool
# ---------------------------------------------------------------------------
HOLDOUT_COLUMNS = ["composition", "label", "A", "B", "family", "year"]


def freeze(allow_dirty: bool = False) -> dict:
    """Crea el holdout congelado, `train_pool` y los folds definitivos. UNA SOLA VEZ.

    Si `holdout_frozen.csv` ya existe, no lo toca: verifica su hash y termina con error.
    Exige el árbol de git limpio (registra el commit), salvo `allow_dirty`.
    """
    import json
    from datetime import UTC, datetime

    import yaml

    from perovskite_pu.splits.build import build_pool, generate_folds
    from perovskite_pu.splits.schemes import temporal_cutoff
    from perovskite_pu.utils.hashing import read_hashes, sha256_file, write_canonical_csv
    from perovskite_pu.utils.paths import (
        CONFIGS_DIR,
        HASHES_FILENAME,
        HOLDOUT_FILENAME,
        INTERIM_DIR,
        PROCESSED_DIR,
        RAW_DIR,
        SPLITS_DIR,
        TRAIN_POOL_FILENAME,
    )  # fmt: skip
    from perovskite_pu.utils.run_metadata import ensure_clean_tree, git_commit
    from perovskite_pu.utils.seeds import load_seed_config

    holdout_path, hashes_path = SPLITS_DIR / HOLDOUT_FILENAME, SPLITS_DIR / HASHES_FILENAME
    if holdout_path.exists():
        ok = read_hashes(hashes_path).get(HOLDOUT_FILENAME) == sha256_file(holdout_path)
        state = "correcto" if ok else "DISTINTO"
        raise FileExistsError(
            f"El holdout ya está congelado ({holdout_path}); hash {state}. No se vuelve a crear."
        )
    ensure_clean_tree(allow_dirty=allow_dirty)
    cfg = yaml.safe_load((CONFIGS_DIR / "data.yaml").read_text("utf-8"))
    seeds = load_seed_config()
    pool = build_pool(
        pd.read_parquet(INTERIM_DIR / "f1_5_labels.parquet"),
        pd.read_parquet(INTERIM_DIR / "f1_5_entries.parquet"),
        pd.read_parquet(RAW_DIR / "mp_abo3_provenance.parquet"),
        cfg["b_families"],
    )
    sel = select_reserved_b(
        pool, seeds.holdout_seed, bounds=tuple(cfg["holdout"]["target_positive_fraction"])
    )
    holdout, train_pool = split_holdout(pool, sel.reserved_b)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    h_sha = write_canonical_csv(holdout[HOLDOUT_COLUMNS], holdout_path, ["composition"])
    train_pool.to_parquet(PROCESSED_DIR / TRAIN_POOL_FILENAME, index=False)

    # Folds definitivos de todos los esquemas y semillas, SOLO sobre train_pool.
    split_cfg = dict(cfg["splits"])
    cutoff = temporal_cutoff(
        train_pool.loc[train_pool["label"] == 1, "year"],
        split_cfg["temporal"]["target_test_fraction"],
    )
    split_cfg["temporal"] = {**split_cfg["temporal"], "cutoff_year": cutoff}
    X = pd.read_parquet(INTERIM_DIR / "magpie.parquet").loc[train_pool["composition"]].to_numpy()
    folds = generate_folds(train_pool, X, split_cfg, cfg["b_families"], seeds.seeds)
    folds_path = SPLITS_DIR / "folds.parquet"
    folds.to_parquet(folds_path, index=False)

    commit = git_commit()
    record = {
        "description": "SHA-256 de los archivos de data/splits/ y metadatos del holdout (F1.8). "
        "El holdout se creó una sola vez; no se lee hasta F4.1 (load_holdout, HOLDOUT_UNLOCK=1).",
        "files": {HOLDOUT_FILENAME: h_sha, "folds.parquet": sha256_file(folds_path)},
        "holdout": {
            "holdout_seed": seeds.holdout_seed, "reserved_b": sel.reserved_b,
            "reserved_b_per_family": sel.per_family, "selection_attempt": sel.attempt,
            "positive_fraction": sel.positive_fraction,
            "n_compositions": len(holdout), "n_positives": int(holdout["label"].sum()),
            "train_pool_compositions": len(train_pool),
            "train_pool_positives": int(train_pool["label"].sum()),
            "train_pool_sha256": sha256_file(PROCESSED_DIR / TRAIN_POOL_FILENAME),
            "code_commit": commit, "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        },
        "folds": {"seeds": list(seeds.seeds), "schemes": sorted(folds["scheme"].unique()),
                  "cluster_k": split_cfg["cluster"]["n_clusters"], "temporal_cutoff_year": cutoff,
                  "code_commit": commit},
    }  # fmt: skip
    with open(hashes_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(record, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return record


def split_frozen(
    pool: pd.DataFrame, reserved_b: Sequence[str], holdout_sha256: str
) -> pd.DataFrame:
    """`train_pool` con los B reservados ya congelados, tras verificar la parte reservada.

    Separa `pool` con `split_holdout` y calcula EN MEMORIA el SHA-256 de la forma canónica de
    la parte reservada (mismas columnas y orden con que `freeze` escribió el CSV). Si no
    coincide con `holdout_sha256`, los datos regenerados no son los congelados y se aborta.
    No lee ni escribe el CSV congelado, y la parte reservada no se devuelve ni se muestra
    (decisión de la autora, 2026-11-15).
    """
    from perovskite_pu.utils.hashing import canonical_csv_bytes, sha256_bytes

    holdout, train_pool = split_holdout(pool, reserved_b)
    got = sha256_bytes(canonical_csv_bytes(holdout[HOLDOUT_COLUMNS], ["composition"]))
    del holdout  # no se conserva más allá del hash
    if got != holdout_sha256:
        raise ValueError(
            "La parte reservada regenerada no coincide con el hash congelado "
            f"({got[:12]}… frente a {holdout_sha256[:12]}…). No se escribe train_pool."
        )
    return train_pool


def rebuild_train_pool() -> dict:
    """Regenera `train_pool.parquet` desde los datos intermedios, SIN volver a congelar.

    Pensado para un contenedor nuevo, donde `data/processed/` está vacío. Construye el pool
    como `freeze`, lo separa con los B reservados registrados en `HASHES.json` (no los vuelve a
    sortear) y verifica el hash de la parte reservada con `split_frozen`. Escribe primero un
    archivo provisional y solo lo deja como `train_pool.parquet` si su SHA-256 coincide con
    `train_pool_sha256`; si no, lo borra y aborta. No toca `holdout_frozen.csv`,
    `folds.parquet` ni `HASHES.json`.
    """
    import json
    import os

    import yaml

    from perovskite_pu.splits.build import build_pool
    from perovskite_pu.utils.hashing import sha256_file
    from perovskite_pu.utils.paths import (
        CONFIGS_DIR,
        HASHES_FILENAME,
        HOLDOUT_FILENAME,
        INTERIM_DIR,
        PROCESSED_DIR,
        RAW_DIR,
        ROOT,
        SPLITS_DIR,
        TRAIN_POOL_FILENAME,
    )  # fmt: skip

    record = json.loads((SPLITS_DIR / HASHES_FILENAME).read_text("utf-8"))
    frozen = record["holdout"]
    cfg = yaml.safe_load((CONFIGS_DIR / "data.yaml").read_text("utf-8"))
    pool = build_pool(
        pd.read_parquet(INTERIM_DIR / "f1_5_labels.parquet"),
        pd.read_parquet(INTERIM_DIR / "f1_5_entries.parquet"),
        pd.read_parquet(RAW_DIR / "mp_abo3_provenance.parquet"),
        cfg["b_families"],
    )
    train_pool = split_frozen(pool, frozen["reserved_b"], record["files"][HOLDOUT_FILENAME])

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    final = PROCESSED_DIR / TRAIN_POOL_FILENAME
    tmp = final.with_name(final.name + ".tmp")
    train_pool.to_parquet(tmp, index=False)
    got = sha256_file(tmp)
    if got != frozen["train_pool_sha256"]:
        tmp.unlink()
        raise ValueError(
            f"train_pool regenerado con otro hash ({got[:12]}… frente a "
            f"{frozen['train_pool_sha256'][:12]}…); no se escribe."
        )
    os.replace(tmp, final)
    return {
        "train_pool": final.relative_to(ROOT).as_posix(),
        "compositions": len(train_pool),
        "positives": int(train_pool["label"].sum()),
        "train_pool_sha256": got,
        "holdout_hash": "coincide (verificado en memoria)",
    }


def main(argv: list[str] | None = None) -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="F1.8: congela el holdout (una sola vez) o regenera train_pool sin congelar."
    )
    parser.add_argument("command", choices=["freeze", "rebuild"])
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "rebuild":
        print(json.dumps(rebuild_train_pool(), indent=2, ensure_ascii=False))
        return
    record = freeze(allow_dirty=args.allow_dirty)
    print(
        json.dumps(
            {k: record[k] for k in ("files", "holdout", "folds")}, indent=2, ensure_ascii=False
        )
    )


if __name__ == "__main__":  # requisito en Windows (regla 2.2)
    main()
