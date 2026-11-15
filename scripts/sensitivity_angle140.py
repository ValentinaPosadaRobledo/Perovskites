"""F3.6, sensibilidad 5a: ángulo B–O–B mínimo de 140° en lugar de 135° (preregistro §10).

Reaplica a las entradas guardadas de F1.5 la regla automática de `data.identify` con el umbral
de sensibilidad: una entrada que era `perovskite` con un ángulo B–O–B mínimo por debajo de
140° pasa a `perovskite_borderline` (igual que haría `identify` con ese umbral). Las entradas
revisadas a mano (F1.4) conservan su veredicto. Después se reetiqueta cada composición con
`label_composition` (la misma función que F1.5) y se listan las positivas de `train_pool` que
dejan de serlo (pasan a ambiguas: quedan fuera de P y U).

Escribe `data/splits/sensitivity/angle140_lost_positives.csv` (solo composiciones, en forma
canónica). Las rutas se pueden pasar como argumentos para leer los datos de otra copia del
repositorio sin escribir en ella.

Uso:
    uv run python scripts/sensitivity_angle140.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from perovskite_pu.data.labeling import lost_positives, reclassify_min_angle
from perovskite_pu.utils.paths import ROOT


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--entries", type=Path, default=ROOT / "data/interim/f1_5_entries.parquet")
    ap.add_argument("--pool", type=Path, default=ROOT / "data/processed/train_pool.parquet")
    ap.add_argument("--config", type=Path, default=ROOT / "configs/data.yaml")
    out_default = ROOT / "data/splits/sensitivity/angle140_lost_positives.csv"
    ap.add_argument("--out", type=Path, default=out_default)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text("utf-8"))
    conn = cfg["identification"]["octahedral_connectivity"]
    angle = float(conn["sensitivity_min_b_o_b_angle_deg"])
    entries = pd.read_parquet(args.entries)
    pool = pd.read_parquet(args.pool)
    positives = pool.loc[pool["label"] == 1, "composition"]
    lost = lost_positives(entries, positives, angle).sort_values("composition", ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Forma canónica: ordenado, LF, UTF-8.
    lost.to_csv(args.out, index=False, lineterminator="\n", encoding="utf-8")
    n_flip = int(reclassify_min_angle(entries, angle)["flipped_angle"].sum())
    print(f"Umbral {conn['min_b_o_b_angle_deg']}° → {angle}°: {n_flip} entradas pasan a "
          f"borderline; {len(lost)} de {int((pool['label'] == 1).sum())} P de train_pool dejan "
          f"de ser positivas → {args.out}")  # fmt: skip
    print(lost.to_string(index=False))


if __name__ == "__main__":  # requisito en Windows (regla 2.2)
    main()
