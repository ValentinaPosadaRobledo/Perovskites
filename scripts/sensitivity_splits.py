"""F3.6: particiones de las sensibilidades 6, 12 y 13 (preregistro §10), sin tocar las congeladas.

- **6**: `loeo_B` sin fusionar los elementos B con < 5 positivas (`report_separately`).
- **12**: `loeo_A` y `loeo_B` con los sitios asignados por tamaño (`A_size`, `B_size`; r_A > r_B)
  en lugar de por estructura, donde la regla de tamaño da una asignación.
- **13**: `train_pool` más las composiciones ambiguas como positivas (las de elemento B no
  reservado), con folds nuevos de todos los esquemas salvo `temporal` (las ambiguas no tienen
  año de primera caracterización). Escribe además ese pool aumentado.

Mismo código y semillas que F1.8 (`splits.build.generate_folds`, `configs/seeds.yaml`); la
matriz MAGPIE de `cluster` es la misma de F1.8. El pool aumentado no comparte composiciones
con el conjunto reservado: se comprueba con los elementos B reservados de `HASHES.json`.

Salidas (solo composiciones en `data/splits/sensitivity/`; el pool aumentado, que lleva
etiquetas y banderas, en `data/processed/sensitivity/`, no versionado) y sus SHA-256 en
`data/splits/sensitivity/HASHES.json`.

Uso:
    uv run python scripts/sensitivity_splits.py [--data-root <otra copia>] [--out-root <destino>]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from perovskite_pu.splits.build import generate_folds, pool_with_ambiguous_as_positive
from perovskite_pu.splits.schemes import SCHEMES
from perovskite_pu.utils.hashing import sha256_file
from perovskite_pu.utils.paths import ROOT
from perovskite_pu.utils.seeds import load_seed_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path, default=ROOT, help="copia de la que se leen datos")
    ap.add_argument("--out-root", type=Path, default=ROOT, help="copia en la que se escribe")
    args = ap.parse_args()
    dr, out_root = args.data_root.resolve(), args.out_root.resolve()
    cfg = yaml.safe_load((ROOT / "configs/data.yaml").read_text("utf-8"))
    seeds = load_seed_config().seeds
    fam = cfg["b_families"]
    pool = pd.read_parquet(dr / "data/processed/train_pool.parquet")
    magpie = pd.read_parquet(dr / "data/interim/magpie.parquet")
    labels = pd.read_parquet(dr / "data/interim/f1_5_labels.parquet")
    hashes_holdout = json.loads((dr / "data/splits/HASHES.json").read_text("utf-8"))["holdout"]
    reserved = set(hashes_holdout["reserved_b"])
    base_cfg = dict(cfg["splits"])
    # Mismo corte temporal que F1.8 (no se usa aquí, pero make_folds lo exige para `temporal`).
    splits_dir = out_root / "data/splits/sensitivity"
    proc_dir = out_root / "data/processed/sensitivity"
    splits_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)
    X = magpie.loc[pool["composition"]].to_numpy()
    outputs = {}

    # 6: loeo_B sin fusionar grupos pequeños.
    cfg6 = {**base_cfg, "loeo_B": {**base_cfg["loeo_B"], "small_group_policy": "report_separately"}}
    f6 = generate_folds(pool, X, cfg6, fam, seeds, ["loeo_B"])
    outputs["s06_loeo_B_unmerged.parquet"] = f6

    # 12: sitios por tamaño donde la regla da asignación (si no, se conserva la estructural).
    p12 = pool.copy()
    has = p12["A_size"].notna() & p12["B_size"].notna()
    p12.loc[has, "A"], p12.loc[has, "B"] = p12.loc[has, "A_size"], p12.loc[has, "B_size"]
    f12 = generate_folds(p12, X, base_cfg, fam, seeds, ["loeo_A", "loeo_B"])
    outputs["s12_sites_by_size.parquet"] = f12

    # 13: ambiguas como positivas; folds nuevos de todos los esquemas salvo `temporal`.
    p13 = pool_with_ambiguous_as_positive(pool, labels, reserved, fam)
    X13 = magpie.loc[p13["composition"]].to_numpy()
    f13 = generate_folds(p13, X13, base_cfg, fam, seeds, [s for s in SCHEMES if s != "temporal"])
    outputs["s13_ambiguous_as_positive.parquet"] = f13
    pool13 = proc_dir / "s13_ambiguous_as_positive_pool.parquet"
    p13.to_parquet(pool13, index=False)

    # Lista de {ruta, sha256}: formato que reconoce el filtro de detect-secrets.
    hashes = []
    for name, table in outputs.items():
        path = splits_dir / name
        table.to_parquet(path, index=False)
        hashes.append({"path": name, "sha256": sha256_file(path)})
    hashes.append({"path": "../../processed/sensitivity/" + pool13.name,
                   "sha256": sha256_file(pool13)})  # fmt: skip
    (splits_dir / "HASHES.json").write_text(json.dumps(hashes, indent=2) + "\n", "utf-8")
    print(f"6: {f6['fold'].nunique()} folds de loeo_B (semilla 0: "
          f"{f6[f6.seed == 0]['fold'].nunique()}); 12: {has.sum()} composiciones con sitios por "
          f"tamaño, de ellas {(p12['A'] != pool['A']).sum()} cambian A; 13: "
          f"{int(p13['added_ambiguous'].sum())} ambiguas añadidas como P "
          f"({int(p13['label'].sum())} P y {int((p13['label'] == 0).sum())} U)")  # fmt: skip


if __name__ == "__main__":  # requisito en Windows (regla 2.2)
    main()
