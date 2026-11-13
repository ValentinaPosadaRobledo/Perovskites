"""F1.4: prepara la planilla de validación manual de casos límite.

Selecciona de forma DETERMINISTA (sin azar) unos 60 casos de las categorías de
`docs/validacion_manual.md` y escribe `docs/validacion_manual_casos.csv` con:
- la clasificación automática de F1.3 y sus métricas (NC, ángulo, compartición...);
- IDs de MP en ambos formatos (nuevo `mp-aaaa...` y antiguo numérico) para buscarlos;
- columnas VACÍAS para la revisora: clasificación manual, coincide, motivo, revisora, fecha.

No incluye coordenadas atómicas (licencia ICSD): solo IDs y descriptores derivados.

Uso:
    uv run python scripts/prepare_f1_4.py
"""

from __future__ import annotations

import pandas as pd
import yaml
from pymatgen.core import Composition

from perovskite_pu.data.external import (
    BARTEL_DIR,
    gu_root,
    label_conflicts,
    load_bartel2019,
    load_gu2022,
    to_legacy_mpid,
)
from perovskite_pu.utils.paths import CONFIGS_DIR, INTERIM_DIR, RAW_DIR, ROOT

OUT = ROOT / "docs" / "validacion_manual_casos.csv"
CORNER = ["perovskite", "perovskite_borderline"]


def pick(df: pd.DataFrame, n: int, taken: set[str], by_formula: bool = True) -> pd.DataFrame:
    """Toma las primeras `n` filas no elegidas antes (una por composición si `by_formula`)."""
    df = df[~df["material_id"].isin(taken)]
    if by_formula:
        df = df.drop_duplicates("reduced_formula")
    out = df.head(n)
    taken.update(out["material_id"])
    return out


def same_comp(a: str, b: str) -> bool:
    return Composition(a).reduced_composition == Composition(b).reduced_composition


def main() -> None:
    cfg = yaml.safe_load((CONFIGS_DIR / "data.yaml").read_text("utf-8"))
    t = pd.read_parquet(INTERIM_DIR / "f1_3_identification.parquet").merge(
        pd.read_parquet(RAW_DIR / "mp_abo3_index.parquet"), on="material_id"
    )
    t["experimental"] = ~t["theoretical"].astype(bool)
    e = t[t["experimental"]].copy()
    thr = cfg["identification"]["octahedral_connectivity"]["min_b_o_b_angle_deg"]
    taken: set[str] = set()
    blocks: list[tuple[str, str, pd.DataFrame]] = []

    def add(cat: str, hint: str, rows: pd.DataFrame) -> None:
        blocks.append((cat, hint, rows))

    # 1. Perovskitas muy inclinadas cerca del umbral de ángulo (ambos lados).
    near = e[e["cls"].isin(CORNER) & e["bob_angle_min"].between(thr - 5, thr + 10)].copy()
    near["d"] = (near["bob_angle_min"] - thr).abs()
    add("1 inclinada cerca del umbral", "¿Octaedros BO6 que comparten solo vértices? ¿Es una perovskita Pnma/R-3c muy inclinada?",
        pick(near.sort_values(["d", "material_id"]), 8, taken))  # fmt: skip
    # 2. Politipos hexagonales: primero los mixtos (vértices + caras: 4H/6H), luego 2H.
    hexa = e[e["cls"] == "hexagonal_polytype"].copy()
    hexa["mixed"] = hexa["n_corner_links"] > 0
    add("2 politipo hexagonal", "¿Comparten caras? ¿2H (solo caras) o 4H/6H (caras + vértices)?",
        pick(hexa.sort_values(["mixed", "material_id"], ascending=[False, True]), 6, taken))  # fmt: skip
    # 3. Ambos cationes octaédricos: ilmenita (aristas) y tipo LiNbO3 (vértices, A con NC 6).
    both = e[e["reasons"].str.contains("ambos cationes octaédricos")].copy()
    add("3 ilmenita (aristas)", "Ilmenita/corindón: ¿octaedros que comparten aristas y caras? No es perovskita.",
        pick(both[both["n_edge_links"] > 0].sort_values("material_id"), 3, taken))  # fmt: skip
    add("3 tipo LiNbO3", "¿Tipo LiNbO3 (A en sitio octaédrico) o perovskita con A de par libre?",
        pick(both[both["max_shared_o"] == 1].sort_values("material_id"), 3, taken))  # fmt: skip
    # 4. B no octaédrico (p. ej. YMnO3 hexagonal, bipirámides).
    add("4 B no octaédrico", "¿Ningún catión octaédrico (bipirámide, tetraedro)? No es perovskita.",
        pick(e[e["reasons"].str.contains("ningún catión es octaédrico")].sort_values("material_id"), 4, taken))  # fmt: skip
    # 5. Aristas compartidas con A grande (post-perovskita, CaIrO3-tipo).
    post = e[(e["cls"] == "non_perovskite") & (e["n_edge_links"] > 0) & (e["a_cn_min"] >= 7)]
    add("5 aristas con A grande", "¿Post-perovskita (capas de octaedros con aristas y vértices)?",
        pick(post.sort_values("material_id"), 3, taken))  # fmt: skip
    # 6. Polianiones: un caso por formador distinto.
    poly = e[e["cls"] == "polyanion"].sort_values(["b_element", "material_id"])
    add("6 polianión", "¿Grupo XO3 aislado (carbonato, nitrato, yodato…)? Excluido por definición.",
        pick(poly.drop_duplicates("b_element"), 6, taken))  # fmt: skip
    # 7. Oxidación ambigua: primero las de la lista de configs, luego perovskitas ambiguas.
    amb_list = cfg["sensitivity_flags"]["ambiguous_oxidation_state"]
    amb = e[e["oxi_ambiguous"]].copy()
    amb["listed"] = amb["reduced_formula"].map(lambda f: any(same_comp(f, x) for x in amb_list))
    amb = amb[amb["listed"] | (amb["cls"] == "perovskite")]
    add("7 oxidación ambigua", "¿Estados de oxidación razonables? ¿Cambia la clasificación?",
        pick(amb.sort_values(["listed", "material_id"], ascending=[False, True]), 5, taken))  # fmt: skip
    # 8. Alta presión / metaestables de la lista de configs.
    hp = cfg["sensitivity_flags"]["high_pressure_or_metastable"]
    hpe = e[
        e["reduced_formula"].map(lambda f: any(same_comp(f, x) for x in hp)) & e["cls"].isin(CORNER)
    ]
    add("8 alta presión / metaestable", "¿Polimorfo de alta presión? ¿Cuenta como positivo (con bandera)?",
        pick(hpe.sort_values("material_id"), 4, taken))  # fmt: skip
    # 9. Desacuerdo con el prototipo cúbico (StructureMatcher).
    add("9 perovskita sin coincidir con prototipo", "La conectividad dice perovskita y StructureMatcher no: ¿quién acierta?",
        pick(e[(e["cls"] == "perovskite") & (e["matches_cubic_prototype"] == False)].sort_values("material_id"), 3, taken))  # noqa: E712  # fmt: skip
    add("9 no perovskita que coincide con prototipo", "StructureMatcher dice perovskita y la conectividad no: ¿quién acierta?",
        pick(e[~e["cls"].isin(CORNER) & (e["matches_cubic_prototype"] == True)].sort_values("material_id"), 3, taken))  # noqa: E712  # fmt: skip
    # 10. NC(A) baja como único motivo de caso límite.
    low_a = e[(e["cls"] == "perovskite_borderline") & ~e["reasons"].str.contains("ángulo")]
    add("10 NC(A) baja", "¿El sitio A está en la cavidad cuboctaédrica (perovskita) o en otro entorno?",
        pick(low_a.sort_values("a_cn_min"), 5, taken))  # fmt: skip
    # 11. Conflictos con fuentes externas (Bartel, Gu): los de mayor interés químico.
    bartel_df = load_bartel2019(BARTEL_DIR / cfg["external_sources"]["bartel2019"]["table"])
    gu = load_gu2022(gu_root(), cfg["external_sources"]["gu2022"]["split_index"])
    conf = set(label_conflicts(bartel_df, gu)["reduced_formula"]) | {
        "SrSiO3",
        "TiMnO3",
        "LaYO3",
        "ScCrO3",
        "ScBiO3",
        "AlBiO3",
    }
    ext = e[
        e["reduced_formula"].isin(conf)
        & e["cls"].isin([*CORNER, "non_perovskite", "hexagonal_polytype"])
    ]
    add("11 conflicto con Bartel/Gu", "Las fuentes discrepan: ¿qué es realmente esta estructura?",
        pick(ext.sort_values(["reduced_formula", "material_id"]), 8, taken))  # fmt: skip

    rows = []
    for cat, hint, df in blocks:
        for _, r in df.iterrows():
            rows.append({
                "caso": len(rows) + 1, "categoria": cat,
                "formula_ABO3": f"{r['a_element']}{r['b_element']}O3" if r["cls"] != "polyanion" else r["formula_pretty"],
                "material_id": r["material_id"], "mp_id_antiguo": to_legacy_mpid(r["material_id"]),
                "grupo_espacial": r["spacegroup_symbol"], "A": r["a_element"], "B": r["b_element"],
                "oxidacion_A_B": "" if pd.isna(r["oxi_a"]) else f"{r['oxi_a']:+.1f}/{r['oxi_b']:+.1f}",
                "oxidacion_ambigua": bool(r["oxi_ambiguous"]),
                "NC_A": "" if pd.isna(r["a_cn_min"]) else f"{int(r['a_cn_min'])}–{int(r['a_cn_max'])}",
                "O_a_3p3A_del_A": "" if pd.isna(r["a_o_within_3p3"]) else int(r["a_o_within_3p3"]),
                "comparticion_max": {1: "vértice", 2: "arista", 3: "cara"}.get(r["max_shared_o"], ""),
                "angulo_BOB_min": "" if pd.isna(r["bob_angle_min"]) else round(float(r["bob_angle_min"]), 1),
                "coincide_prototipo": r["matches_cubic_prototype"],
                "clasificacion_automatica": r["cls"], "motivos_automaticos": r["reasons"],
                "que_mirar": hint,
                # --- columnas para la revisora ---
                "clasificacion_manual": "", "coincide": "", "motivo_referencia": "", "revisora": "", "fecha": "",
            })  # fmt: skip
    out = pd.DataFrame(rows)
    # utf-8-sig: Excel muestra bien las tildes al abrirlo directamente.
    out.to_csv(OUT, index=False, encoding="utf-8-sig", lineterminator="\n")
    print(out.groupby("categoria").size().to_string())
    print(f"\nTotal: {len(out)} casos -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":  # requisito en Windows (regla 2.2)
    main()
