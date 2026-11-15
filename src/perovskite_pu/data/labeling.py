"""F1.5: deduplicación, etiquetado por composición y banderas de sensibilidad.

Entrada: la descarga de MP (F1.1), la identificación por conectividad (F1.3) y la revisión
manual (F1.4). Salida: una fila por composición reducida con su etiqueta y sus banderas.

Etiquetas (sección 1 de CLAUDE.md y decisiones de la bitácora):
    positive                    alguna entrada EXPERIMENTAL (`theoretical == False`) es perovskita
    unlabeled                   ninguna entrada experimental (los U NO son negativos)
    synthesized_non_perovskite  experimental, pero ninguna entrada es perovskita ni ambigua
    ambiguous                   experimental; su única evidencia es un caso límite sin revisar o
                                "dudoso": fuera de P y U (sensibilidad: P)
    polyanion                   todas sus entradas son polianiones: fuera del conjunto

La revisión manual de F1.4 prevalece sobre la clase automática en las entradas revisadas.

Notas:
- Deduplicación estructural: MP ya entrega un `material_id` por estructura única, pero se
  verifica con StructureMatcher dentro de cada composición (regla 2.1.2); los duplicados se
  cuentan y se conserva un representante.
- "Agrupar composiciones dopadas o casi idénticas con su padre" no aplica: MP solo contiene
  ABO3 ordenados y estequiométricos, así que no hay dopados.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pymatgen.core import Composition, Species, Structure

from perovskite_pu.data.identify import make_matcher
from perovskite_pu.utils.paths import CONFIGS_DIR, INTERIM_DIR, RAW_DIR, ROOT

LABELS = ("positive", "unlabeled", "synthesized_non_perovskite", "ambiguous", "polyanion")
MANUAL_TO_CLS = {
    "perovskita": "perovskite",
    "no_perovskita": "non_perovskite",
    "politipo_hexagonal": "hexagonal_polytype",
    "polianion": "polyanion",
    "dudosa": "doubtful",
}


# ---------------------------------------------------------------------------
# Funciones puras
# ---------------------------------------------------------------------------
def apply_manual_review(entries: pd.DataFrame, review: pd.DataFrame) -> pd.DataFrame:
    """Añade `cls_final`: la clase manual en las entradas revisadas, la automática en el resto."""
    manual = review.set_index("material_id")["clasificacion_manual"].map(MANUAL_TO_CLS)
    if manual.isna().any():
        raise ValueError("Hay veredictos manuales con valores no reconocidos.")
    out = entries.copy()
    out["manual_reviewed"] = out["material_id"].isin(manual.index)
    out["cls_final"] = [
        manual.get(m, c) for m, c in zip(out["material_id"], out["cls"], strict=True)
    ]
    return out


def label_composition(group: pd.DataFrame) -> str:
    """Etiqueta de una composición a partir de sus entradas.

    Usa las columnas `cls_final` (clase tras la revisión manual) y `experimental`.
    """
    if (group["cls_final"] == "polyanion").all():
        return "polyanion"
    exp = group[group["experimental"]]
    if exp.empty:
        return "unlabeled"
    if (exp["cls_final"] == "perovskite").any():
        return "positive"
    if exp["cls_final"].isin(["doubtful", "perovskite_borderline"]).any():
        return "ambiguous"
    return "synthesized_non_perovskite"


def reclassify_min_angle(entries: pd.DataFrame, min_angle: float) -> pd.DataFrame:
    """`cls_final` con otro umbral de ángulo B–O–B mínimo (sensibilidad 5 del preregistro).

    Reaplica la regla automática de `data.identify`: una entrada `perovskite` con ángulo mínimo
    por debajo de `min_angle` pasa a `perovskite_borderline`. Las entradas revisadas a mano
    (F1.4) conservan su veredicto. Solo sirve para umbrales MÁS estrictos que el principal
    (con uno más laxo harían falta las estructuras). Añade la columna `flipped_angle`.
    """
    out = entries.copy()
    flip = (
        (out["cls"] == "perovskite")
        & (out["bob_angle_min"] < min_angle)
        & ~out["manual_reviewed"].astype(bool)
    )
    out.loc[flip, "cls_final"] = "perovskite_borderline"
    out["flipped_angle"] = flip
    return out


def lost_positives(
    entries: pd.DataFrame, positives: Iterable[str], min_angle: float
) -> pd.DataFrame:
    """Positivas que dejan de serlo con el umbral `min_angle`, con su nueva etiqueta."""
    labels = (
        reclassify_min_angle(entries, min_angle)
        .groupby("reduced_formula")
        .apply(label_composition, include_groups=False)
    )
    changed = [(c, labels.get(c)) for c in positives if labels.get(c) != "positive"]
    return pd.DataFrame(changed, columns=["composition", "label_sensitivity"])


def site_assignment(group: pd.DataFrame) -> tuple[str | None, str | None]:
    """Elementos A y B de la composición: los de sus entradas perovskita (experimentales primero).

    Si ninguna entrada es perovskita, se usa la asignación más frecuente entre sus entradas.
    """
    for subset in (
        group[(group["cls_final"] == "perovskite") & group["experimental"]],
        group[group["cls_final"] == "perovskite"],
        group,
    ):
        pairs = subset.dropna(subset=["a_element", "b_element"])
        if not pairs.empty:
            a, b = pairs.groupby(["a_element", "b_element"]).size().idxmax()
            return a, b
    return None, None


def charge_balanced(formula: str) -> bool:
    """True si la composición admite balance de carga con los estados de oxidación de ICSD.

    Usa `Composition.oxi_state_guesses`, que por defecto toma los estados OBSERVADOS EN ICSD
    (`Element.icsd_oxidation_states`) y, si un elemento no los tiene, los comunes
    (`common_oxidation_states`). Es más permisivo que los comunes solos: admite Mn3+, así que
    LaMnO3 equilibra la carga (con los comunes, no). Ojo: aun así falla en
    compuestos reales con estados poco comunes (Fe4+ en CaFeO3, Bi mixto en BaBiO3), así que
    se usa como BANDERA y como línea base, nunca para excluir.
    """
    return bool(Composition(formula).oxi_state_guesses())


def size_rule_sites(formula: str) -> tuple[str | None, str | None]:
    """Sitios por tamaño (sensibilidad): A es el catión con mayor radio de Shannon en VI.

    Convención de Bartel et al. (2019), r_A > r_B. Usa la conjetura de oxidación más probable;
    devuelve (None, None) si no hay balance de carga o falta algún radio.
    """
    comp = Composition(formula)
    guesses = comp.oxi_state_guesses()
    if not guesses:
        return None, None
    ox = guesses[0]
    radii = {}
    for el in (e.symbol for e in comp.elements if e.symbol != "O"):
        try:
            sp = Species(el, ox[el])
            try:
                radii[el] = sp.get_shannon_radius(cn="VI", radius_type="ionic")
            except (KeyError, ValueError):
                radii[el] = sp.ionic_radius
        except (KeyError, ValueError):
            return None, None
        if radii[el] is None:
            return None, None
    a, b = sorted(radii, key=lambda e: -radii[e])
    return a, b


def matches_any(formula: str, formulas: Iterable[str]) -> bool:
    """True si `formula` tiene la misma composición reducida que alguna de `formulas`."""
    comp = Composition(formula).reduced_composition
    return any(comp == Composition(f).reduced_composition for f in formulas)


def metastable_gap(group: pd.DataFrame) -> float | None:
    """E_hull de la perovskita experimental más estable menos la del polimorfo más estable.

    Devuelve None si no hay perovskita experimental con E_hull o si el estado fundamental de la
    composición es una perovskita (entonces no es metaestable). Usa la columna `ehull`.
    """
    g = group.dropna(subset=["ehull"])
    perov = g[(g["cls_final"] == "perovskite") & g["experimental"]]
    if perov.empty:
        return None
    ground = g.loc[g["ehull"].idxmin()]
    if ground["cls_final"] == "perovskite":
        return 0.0
    return float(perov["ehull"].min() - ground["ehull"])


def build_labels(
    entries: pd.DataFrame, flags_cfg: dict[str, Any], bartel_non_perovskite: set[str]
) -> pd.DataFrame:
    """Una fila por composición con etiqueta, sitios A/B, recuentos y banderas.

    Banderas (para los análisis de sensibilidad; NO cambian la etiqueta principal):
        flag_high_pressure         en la lista curada de configs (alta presión o metaestable)
        flag_metastable_dft        su perovskita experimental queda ≥ umbral (eV/átomo) por
                                   encima del polimorfo más estable, que no es perovskita
        flag_ambiguous_oxidation   en la lista de configs, o todas sus entradas perovskita
                                   experimentales tienen oxidación ambigua
        flag_bartel_conflict       positiva aquí pero NO perovskita según Bartel (candidata a
                                   alta presión, a confirmar)
        flag_only_hexagonal        sintetizada solo como politipo hexagonal
        flag_icsd_without_exp      tiene IDs de ICSD pero ninguna entrada `theoretical == False`
        flag_no_charge_balance     sin balance de carga con los estados de ICSD (pymatgen)
        flag_site_rule_disagrees   la asignación A/B por estructura difiere de la de tamaño

    Sitios: `A`, `B` salen de la estructura (principal); `A_size`, `B_size` de la regla de
    tamaño (sensibilidad de loeo_A/loeo_B; decisión de la autora, 2026-11-14).
    """
    high_pressure = flags_cfg.get("high_pressure_or_metastable", [])
    ambiguous_ox = flags_cfg.get("ambiguous_oxidation_state", [])
    gap_thr = flags_cfg.get("metastable_dft_gap_ev_per_atom")
    rows = []
    for comp, g in entries.groupby("reduced_formula"):
        label = label_composition(g)
        a, b = site_assignment(g)
        exp = g[g["experimental"]]
        perov_exp = exp[exp["cls_final"] == "perovskite"]
        gap = metastable_gap(g) if "ehull" in g else None
        a_size, b_size = size_rule_sites(comp)
        rows.append({
            "reduced_formula": comp, "label": label, "A": a, "B": b,
            "A_size": a_size, "B_size": b_size,
            "n_entries": len(g), "n_unique_structures": int(g["dup_group"].nunique()),
            "n_experimental": len(exp), "n_perovskite_experimental": len(perov_exp),
            "manual_reviewed": bool(g["manual_reviewed"].any()),
            "flag_high_pressure": matches_any(comp, high_pressure),
            "metastable_gap_ev": gap,
            "flag_metastable_dft": gap is not None and gap_thr is not None and gap >= gap_thr,
            "flag_ambiguous_oxidation": matches_any(comp, ambiguous_ox)
            or (not perov_exp.empty and bool(perov_exp["oxi_ambiguous"].all())),
            "flag_bartel_conflict": label == "positive" and comp in bartel_non_perovskite,
            "flag_only_hexagonal": label == "synthesized_non_perovskite"
            and bool((exp["cls_final"] == "hexagonal_polytype").all()),
            "flag_icsd_without_exp": bool((g["n_icsd_ids"] > 0).any()) and exp.empty,
            "flag_no_charge_balance": not charge_balanced(comp),
            "flag_site_rule_disagrees": a_size is not None and a is not None and a != a_size,
        })  # fmt: skip
    return pd.DataFrame(rows).sort_values("reduced_formula", ignore_index=True)


# ---------------------------------------------------------------------------
# Deduplicación estructural
# ---------------------------------------------------------------------------
def structural_duplicate_groups(
    entries: pd.DataFrame, structures: dict[str, dict], sm_cfg: dict[str, Any]
) -> pd.Series:
    """Grupo de duplicados estructurales de cada entrada, dentro de su composición.

    Devuelve una Serie indexada por `material_id` con un identificador "formula#k".
    """
    matcher = make_matcher(sm_cfg)
    groups: dict[str, str] = {}
    for comp, g in entries.groupby("reduced_formula"):
        ids = sorted(g["material_id"])
        if len(ids) == 1:
            groups[ids[0]] = f"{comp}#0"
            continue
        structs = [Structure.from_dict(structures[i]) for i in ids]
        for k, cluster in enumerate(matcher.group_structures(structs)):
            for s in cluster:
                groups[ids[structs.index(s)]] = f"{comp}#{k}"
    return pd.Series(groups, name="dup_group")


# ---------------------------------------------------------------------------
# Ejecución
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="F1.5: etiquetado por composición.")
    parser.add_argument("--config", type=Path, default=CONFIGS_DIR / "data.yaml")
    parser.add_argument("--out", type=Path, default=INTERIM_DIR / "f1_5_labels.parquet")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text("utf-8"))

    from perovskite_pu.data.external import BARTEL_DIR, load_bartel2019  # import tardío

    entries = pd.read_parquet(INTERIM_DIR / "f1_3_identification.parquet").merge(
        pd.read_parquet(RAW_DIR / "mp_abo3_index.parquet"), on="material_id"
    )
    entries["experimental"] = ~entries["theoretical"].astype(bool)
    review = pd.read_csv(ROOT / cfg["labeling"]["manual_review"], encoding="utf-8-sig")
    entries = apply_manual_review(entries, review)

    docs = json.loads(gzip.decompress((RAW_DIR / "mp_abo3_summary.json.gz").read_bytes()))
    structures = {str(d["material_id"]): d["structure"] for d in docs}
    dup = structural_duplicate_groups(
        entries, structures, cfg["identification"]["structure_matcher"]
    )
    entries["dup_group"] = entries["material_id"].map(dup)
    thermo = pd.read_parquet(RAW_DIR / "mp_abo3_thermo.parquet")
    gga = thermo[thermo["thermo_type"] == cfg["materials_project"]["thermo_types"][0]]
    entries["ehull"] = entries["material_id"].map(gga.set_index("material_id")["energy_above_hull"])

    bartel = load_bartel2019(BARTEL_DIR / cfg["external_sources"]["bartel2019"]["table"])
    bartel_non = set(bartel.loc[bartel["abo3_oxide"] & ~bartel["is_perovskite"], "reduced_formula"])
    labels = build_labels(entries, cfg["sensitivity_flags"], bartel_non)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(args.out, index=False)
    entries.to_parquet(args.out.with_name("f1_5_entries.parquet"), index=False)
    print(labels["label"].value_counts().to_string())


if __name__ == "__main__":  # requisito en Windows (regla 2.2)
    main()
