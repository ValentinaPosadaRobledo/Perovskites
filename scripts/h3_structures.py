"""F3.5 (H3): estructuras de entrada del modelo de grafos para las composiciones de `train_pool`.

Dos juegos, fijados en el preregistro (§8, H3, y §10b, S3) antes de entrenar ningún grafo:

- **A (confirmatorio):** la entrada perovskita de MP de menor E_hull (GGA/GGA+U) de cada
  composición, la misma de `post_dft` y de la línea base de solo simetría
  (`material_id_gga` de `features_post_dft_meta.parquet`). Solo existe para las
  composiciones elegibles de `post_dft` (193 P y 587 U).
- **S3 (exploratorio):** un prototipo cúbico ideal Pm-3m para TODAS las composiciones: A en
  (0, 0, 0), B en (½, ½, ½) y O en (½, ½, 0), (½, 0, ½) y (0, ½, ½), con a = 2 (r_B + r_O),
  r_B el radio de Shannon que usa t (`bartel_rB`) y r_O = 1,40 Å. Si t no está definido, o si
  la asignación de sitios de t no coincide con la estructural (A, B de `train_pool`), a es la
  mediana de los a definidos (no usa etiquetas).

Escribe en `data/interim/h3/` (no versionado: las estructuras de MP se redistribuyen solo
dentro del grupo, con su licencia CC BY) el paquete del entrenador aislado
(`isolated/gu2022/scripts/h3_train.py`):

- `{A,S3}/<composición>.cif` y `{A,S3}/id_prop.csv` (composición, etiqueta), el formato que
  lee el cargador de Gu;
- `manifest.csv`: composición, etiqueta, material_id, a del prototipo, uso de la mediana y
  SHA-256 de cada CIF;
- `folds.csv`: los folds congelados de F1.8 (`data/splits/folds.parquet`) de los esquemas de
  H3 (`random` y los cuatro confirmatorios), solo composiciones;
- `seeds.json`: las semillas de `configs/seeds.yaml`.

Solo lee `train_pool`, nunca el conjunto reservado.

Uso:
    uv run python scripts/h3_structures.py
"""

from __future__ import annotations

import gzip
import json

import numpy as np
import pandas as pd
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter

from perovskite_pu.utils.hashing import sha256_file
from perovskite_pu.utils.paths import ROOT
from perovskite_pu.utils.seeds import load_seed_config

R_O = 1.40  # radio de Shannon de O²⁻ en NC 6 (Å), el mismo `bartel_rX` de t
OUT = ROOT / "data" / "interim" / "h3"
H3_SCHEMES = ("random", "loeo_A", "loeo_B", "family", "cluster")  # preregistro §8


def prototype(a_el: str, b_el: str, a: float) -> Structure:
    """Perovskita cúbica ideal Pm-3m ABO₃ con parámetro de red `a` (Å)."""
    coords = [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
    return Structure(Lattice.cubic(a), [a_el, b_el, "O", "O", "O"], coords)


def main() -> None:
    pool = pd.read_parquet(ROOT / "data/processed/train_pool.parquet").set_index("composition")
    post = pd.read_parquet(ROOT / "data/interim/features_post_dft_meta.parquet")
    pre = pd.read_parquet(ROOT / "data/interim/features_pre_dft_meta.parquet")
    with gzip.open(ROOT / "data/raw/mp_abo3_summary.json.gz", "rt", encoding="utf-8") as f:
        docs = {d["material_id"]: d for d in json.load(f)}

    # Parámetro de red del prototipo: radios de t donde la asignación de sitios coincide.
    meta = pre.reindex(pool.index)
    same_sites = (meta["bartel_B"] == pool["B"]) & (meta["bartel_A"] == pool["A"])
    a_t = 2 * (meta["bartel_rB"].astype(float) + R_O)
    ok = same_sites & a_t.notna()
    a_median = float(np.median(a_t[ok]))
    a_proto = a_t.where(ok, a_median)

    (OUT / "A").mkdir(parents=True, exist_ok=True)
    (OUT / "S3").mkdir(parents=True, exist_ok=True)
    rows = []
    for comp, row in pool.iterrows():
        rec = {"composition": comp, "label": int(row["label"]), "A": row["A"], "B": row["B"],
               "material_id_A": "", "sha256_A": "", "a_S3": round(float(a_proto[comp]), 4),
               "a_S3_is_median": bool(not ok[comp])}  # fmt: skip
        # A: la entrada de post_dft, si la composición es elegible.
        if comp in post.index and bool(post.at[comp, "eligible_post_dft"]):
            mid = post.at[comp, "material_id_gga"]
            s = Structure.from_dict(docs[mid]["structure"])
            path = OUT / "A" / f"{comp}.cif"
            CifWriter(s).write_file(path)
            rec.update(material_id_A=mid, sha256_A=sha256_file(path))
        # S3: el prototipo cúbico ideal, para todas.
        path = OUT / "S3" / f"{comp}.cif"
        CifWriter(prototype(row["A"], row["B"], rec["a_S3"])).write_file(path)
        rec["sha256_S3"] = sha256_file(path)
        rows.append(rec)
    man = pd.DataFrame(rows)
    man.to_csv(OUT / "manifest.csv", index=False, lineterminator="\n")
    # Etiquetas en el formato de Gu (id_prop.csv sin cabecera), una por juego de estructuras.
    for crit, mask in (("A", man["material_id_A"] != ""), ("S3", man["composition"].notna())):
        man.loc[mask, ["composition", "label"]].to_csv(
            OUT / crit / "id_prop.csv", index=False, header=False, lineterminator="\n"
        )
    # Folds de los esquemas de H3 (solo composiciones) y semillas del proyecto.
    folds = pd.read_parquet(ROOT / "data/splits/folds.parquet")
    folds = folds[folds["scheme"].isin(H3_SCHEMES) & folds["composition"].isin(pool.index)]
    folds[["scheme", "seed", "fold", "composition"]].to_csv(OUT / "folds.csv", index=False,
                                                            lineterminator="\n")  # fmt: skip
    seeds = load_seed_config()
    (OUT / "seeds.json").write_text(json.dumps({"seeds": list(seeds.seeds)}) + "\n", "utf-8")
    has_a = man["material_id_A"] != ""
    n_a = {k: int((has_a & (man.label == v)).sum()) for k, v in (("P", 1), ("U", 0))}
    print(json.dumps({
        "compositions": len(man),
        "A": n_a,
        "S3": {"total": len(man), "a_median": round(a_median, 4),
               "uses_median": int(man["a_S3_is_median"].sum()),
               "a_range": [float(man["a_S3"].min()), float(man["a_S3"].max())]},
        "manifest": (OUT / "manifest.csv").relative_to(ROOT).as_posix(),
    }, indent=2))  # fmt: skip


if __name__ == "__main__":  # requisito en Windows (regla 2.2)
    main()
