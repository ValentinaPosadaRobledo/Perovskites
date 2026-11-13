"""F1.3: identificación de perovskitas ABO3 por conectividad (regla 2.1.1 de CLAUDE.md).

Una estructura es perovskita si sus octaedros BO6 comparten SOLO vértices formando una red
tridimensional, no solo si se parece a un prototipo. Para cada estructura:

1. Estados de oxidación con BVAnalyzer; respaldo: `Composition.oxi_state_guesses`. Si
   discrepan, o si BVAnalyzer falla, la entrada queda marcada como ambigua.
2. Polianiones: si uno de los cationes forma polianiones con O (B, C, N, P, S, Se, Cl, Br,
   I; configurable), la clase es `polyanion` y no se analiza la geometría.
3. Coordinación con CrystalNN (solo enlaces catión–anión). Se prueban las dos asignaciones
   de sitio: el elemento B debe tener 6 vecinos O en TODOS sus sitios.
4. Conectividad: cada O "puentea" octaedros. Para cada par B–B (con su traslación) se
   cuentan los O compartidos: 1 = vértice, 2 = arista, 3 = cara.
5. Clases:
     perovskite            solo vértices, cada O une exactamente 2 B, ángulo B–O–B mínimo
                           ≥ umbral y NC(A) en [min, max]
     perovskite_borderline solo vértices y O con 2 B, pero falla el ángulo o la NC de A
                           (revisión manual, F1.4)
     hexagonal_polytype    algún par de octaedros comparte una cara (2H, 4H, 6H…)
     non_perovskite        aristas compartidas, B no octaédrico, O sin 2 B, etc.
     polyanion             ver punto 2
     failed                error de cálculo (se registra el mensaje)
6. Contraste: coincidencia con el prototipo cúbico (StructureMatcher, anónimo).
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import warnings
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, Structure

from perovskite_pu.utils.paths import CONFIGS_DIR, INTERIM_DIR, RAW_DIR

CLASSES = (
    "perovskite",
    "perovskite_borderline",
    "hexagonal_polytype",
    "non_perovskite",
    "polyanion",
    "failed",
)


@dataclass
class Identification:
    """Resultado de identificar una estructura (una fila de la tabla de F1.3)."""

    cls: str
    a_element: str | None = None
    b_element: str | None = None
    oxi_method: str | None = None  # "bv", "guess" o None
    oxi_a: float | None = None
    oxi_b: float | None = None
    oxi_ambiguous: bool = False
    b_cn_all_6: bool = False
    a_cn_min: int | None = None
    a_cn_max: int | None = None
    max_shared_o: int | None = None  # 1 vértice, 2 arista, 3 cara
    n_face_links: int = 0
    n_edge_links: int = 0
    n_corner_links: int = 0
    o_with_2b_frac: float | None = None
    bob_angle_min: float | None = None
    bob_angle_median: float | None = None
    a_o_within_3p0: int | None = None  # diagnóstico: O a < 3,0 Å del sitio A (mín. entre sitios)
    a_o_within_3p3: int | None = None  # diagnóstico: O a < 3,3 Å del sitio A (mín. entre sitios)
    matches_cubic_prototype: bool | None = None
    reasons: list[str] = field(default_factory=list)

    def as_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["reasons"] = ";".join(self.reasons)
        return row


def _symbol(site: Any) -> str:
    """Símbolo del elemento de un sitio, esté o no decorado con estado de oxidación.

    Con oxidación la especie es un `Species` (tiene `.element`); sin ella es un `Element`.
    """
    specie = site.specie
    return getattr(specie, "element", specie).symbol


# ---------------------------------------------------------------------------
# Estados de oxidación
# ---------------------------------------------------------------------------
def decorate_oxidation(structure: Structure) -> tuple[Structure | None, str | None, bool]:
    """Decora la estructura con estados de oxidación.

    Returns:
        (estructura decorada o None, método usado, ambigua). Es ambigua si BVAnalyzer falla
        o si su resultado no coincide con la conjetura más probable por composición.
    """
    guesses = structure.composition.oxi_state_guesses()
    top_guess = guesses[0] if guesses else None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            decorated = BVAnalyzer().get_oxi_state_decorated_structure(structure)
        bv = _mean_oxidation(decorated)
        ambiguous = top_guess is None or any(
            not math.isclose(bv[el], top_guess.get(el, math.nan), abs_tol=0.25) for el in bv
        )
        return decorated, "bv", ambiguous
    except ValueError:
        if top_guess is None:
            return None, None, True
        decorated = structure.copy()
        decorated.add_oxidation_state_by_element(top_guess)
        return decorated, "guess", True


def _mean_oxidation(decorated: Structure) -> dict[str, float]:
    """Estado de oxidación medio por elemento en una estructura decorada."""
    acc: dict[str, list[float]] = defaultdict(list)
    for site in decorated:
        acc[_symbol(site)].append(float(site.specie.oxi_state))
    return {el: float(np.mean(v)) for el, v in acc.items()}


# ---------------------------------------------------------------------------
# Conectividad de los octaedros
# ---------------------------------------------------------------------------
def octahedral_connectivity(
    structure: Structure, nn_info: dict[int, list[dict]], b_indices: list[int]
) -> dict[str, Any]:
    """Cuenta vértices, aristas y caras compartidas entre octaedros BO6, y ángulos B–O–B.

    `nn_info[i]` es la salida de `CrystalNN.get_nn_info` para el sitio i (con imágenes
    periódicas). Para cada O se reúnen los B que lo rodean como vectores O→B; dos B unidos
    a través de ese O forman un enlace B–B identificado por (i, k, vector B_i→B_k). El número
    de O que comparten el mismo enlace es 1 (vértice), 2 (arista) o 3 (cara).
    """
    b_set = set(b_indices)
    o_to_b: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for i in b_indices:
        for nb in nn_info[i]:
            if _symbol(nb["site"]) != "O":
                continue
            j = nb["site_index"]
            vec_o_to_b = structure[i].coords - nb["site"].coords  # independiente de la imagen
            o_to_b[j].append((i, vec_o_to_b))

    links: Counter = Counter()
    angles: list[float] = []
    for entries in o_to_b.values():
        for a in range(len(entries)):
            for b in range(a + 1, len(entries)):
                (i, ri), (k, rk) = entries[a], entries[b]
                # Vector B_i -> B_k redondeado a 0,1 Å (+0.0 normaliza -0.0). Clave canónica:
                # el mismo enlace visto desde B_k (vector opuesto) debe dar la misma clave.
                d = np.round(rk - ri, 1) + 0.0
                fwd, bwd = (i, k, *d), (k, i, *(-d + 0.0))
                key = min(fwd, bwd)
                links[key] += 1
                cos = np.dot(ri, rk) / (np.linalg.norm(ri) * np.linalg.norm(rk))
                angles.append(math.degrees(math.acos(float(np.clip(cos, -1.0, 1.0)))))

    o_indices = [j for j, s in enumerate(structure) if _symbol(s) == "O"]
    n_with_2b = sum(len(o_to_b.get(j, [])) == 2 for j in o_indices)
    shared = Counter(links.values())
    return {
        "max_shared_o": max(links.values()) if links else 0,
        "n_corner_links": shared.get(1, 0),
        "n_edge_links": shared.get(2, 0),
        "n_face_links": sum(v for c, v in shared.items() if c >= 3),
        "o_with_2b_frac": n_with_2b / len(o_indices) if o_indices else math.nan,
        "bob_angle_min": min(angles) if angles else None,
        "bob_angle_median": float(np.median(angles)) if angles else None,
        "b_set_size": len(b_set),
    }


# ---------------------------------------------------------------------------
# Prototipo cúbico (solo contraste)
# ---------------------------------------------------------------------------
def cubic_prototype() -> Structure:
    """Perovskita cúbica ideal ABO3 (Pm-3m): A en el centro, B en el vértice."""
    return Structure(
        Lattice.cubic(3.9),
        ["Sr", "Ti", "O", "O", "O"],
        [[0.5, 0.5, 0.5], [0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]],
    )


# ---------------------------------------------------------------------------
# Identificación de una estructura
# ---------------------------------------------------------------------------
def identify(
    structure: Structure,
    cfg: dict[str, Any],
    cnn: CrystalNN | None = None,
    matcher: StructureMatcher | None = None,
) -> Identification:
    """Clasifica una estructura ABO3 según la sección `identification` de configs/data.yaml."""
    elements = sorted({_symbol(s) for s in structure if _symbol(s) != "O"})
    if len(elements) != 2:
        return Identification("failed", reasons=[f"cationes inesperados: {elements}"])

    polyanion = sorted(set(elements) & set(cfg["polyanion_b_elements"]))
    if polyanion:
        b_el = polyanion[0]
        a_el = next(e for e in elements if e != b_el)
        return Identification("polyanion", a_el, b_el, reasons=[f"formador de polianión: {b_el}"])

    decorated, method, ambiguous = decorate_oxidation(structure)
    cnn = cnn or CrystalNN(cation_anion=decorated is not None)
    target = decorated if decorated is not None else structure
    oxi = _mean_oxidation(decorated) if decorated is not None else {}

    # Vecinos de cada sitio catiónico (los de O no hacen falta).
    nn_info: dict[int, list[dict]] = {}
    for i, site in enumerate(target):
        if _symbol(site) != "O":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                nn_info[i] = cnn.get_nn_info(target, i)

    def o_cn(i: int) -> int:
        return sum(_symbol(nb["site"]) == "O" for nb in nn_info[i])

    def total_cn(i: int) -> int:
        return len(nn_info[i])

    sites_of = {el: [i for i, s in enumerate(target) if _symbol(s) == el] for el in elements}
    # El elemento B: octaédrico (6 vecinos, todos O) en TODOS sus sitios.
    octa = [el for el in elements if all(o_cn(i) == 6 == total_cn(i) for i in sites_of[el])]
    coord = cfg["coordination"]
    result = Identification("non_perovskite", oxi_method=method, oxi_ambiguous=ambiguous)
    if not octa:
        result.reasons.append("ningún catión es octaédrico en todos sus sitios")
        a_el, b_el = elements
    else:
        if len(octa) == 2:
            # Ambos parecen octaédricos (ilmenita, LiNbO3, o un catión con par libre como el
            # Bi3+ de BiFeO3). B es el que forma una red de octaedros que comparten SOLO
            # vértices con cada O uniendo 2 B; si ambos o ninguno, el de mayor carga.
            # Clave de orden: primero la red de solo vértices con cada O uniendo 2 B; luego
            # la red MENOS doblada (mayor ángulo B–O–B mínimo: el Bi3+ de BiFeO3 también
            # forma una "red" de vértices, pero a ~113°); por último, la mayor carga.
            def network_key(el: str) -> tuple:
                c = octahedral_connectivity(target, nn_info, sites_of[el])
                corner = c["max_shared_o"] == 1 and math.isclose(c["o_with_2b_frac"], 1.0)
                return (not corner, -(c["bob_angle_min"] or 0.0), -oxi.get(el, 0.0))

            octa.sort(key=network_key)
            result.reasons.append("ambos cationes octaédricos")
        b_el = octa[0]
        a_el = next(e for e in elements if e != b_el)
        result.b_cn_all_6 = True

    result.a_element, result.b_element = a_el, b_el
    result.oxi_a, result.oxi_b = oxi.get(a_el), oxi.get(b_el)
    a_cns = [o_cn(i) for i in sites_of[a_el]]
    result.a_cn_min, result.a_cn_max = min(a_cns), max(a_cns)
    for cutoff, attr in ((3.0, "a_o_within_3p0"), (3.3, "a_o_within_3p3")):
        counts = [
            sum(_symbol(n) == "O" for n in target.get_neighbors(target[i], cutoff))
            for i in sites_of[a_el]
        ]
        setattr(result, attr, min(counts))

    if result.b_cn_all_6:
        conn = octahedral_connectivity(target, nn_info, sites_of[b_el])
        for k in ("max_shared_o", "n_corner_links", "n_edge_links", "n_face_links",
                  "o_with_2b_frac", "bob_angle_min", "bob_angle_median"):  # fmt: skip
            setattr(result, k, conn[k])
        if conn["n_face_links"] > 0:
            result.cls = "hexagonal_polytype"
            result.reasons.append("octaedros que comparten caras")
        elif conn["n_edge_links"] > 0:
            result.reasons.append("octaedros que comparten aristas")
        elif not math.isclose(conn["o_with_2b_frac"], 1.0):
            result.reasons.append("no todos los O unen exactamente 2 B")
        else:
            angle_ok = (
                conn["bob_angle_min"] >= cfg["octahedral_connectivity"]["min_b_o_b_angle_deg"]
            )
            a_ok = (
                coord["a_site_cn_min"]
                <= result.a_cn_min
                <= result.a_cn_max
                <= coord["a_site_cn_max"]
            )
            # Regla híbrida (decisión de la autora, 2026-11-13): CrystalNN subestima la NC
            # de los cationes A con par libre (Bi3+, Pb2+…). Para ellos basta con tener al
            # menos `lone_pair_min_o` oxígenos a menos de `lone_pair_cutoff_ang`.
            lp = coord.get("lone_pair_a_rule")
            if not a_ok and lp and a_el in lp["elements"]:
                n_close = min(
                    sum(
                        _symbol(n) == "O" for n in target.get_neighbors(target[i], lp["cutoff_ang"])
                    )
                    for i in sites_of[a_el]
                )
                if n_close >= lp["min_o"]:
                    a_ok = True
                    result.reasons.append(f"A con par libre: {n_close} O a < {lp['cutoff_ang']} Å")
            if angle_ok and a_ok:
                result.cls = "perovskite"
            else:
                result.cls = "perovskite_borderline"
                if not angle_ok:
                    result.reasons.append(f"ángulo B–O–B mínimo {conn['bob_angle_min']:.1f}°")
                if not a_ok:
                    result.reasons.append(f"NC(A) {result.a_cn_min}–{result.a_cn_max}")

    if matcher is not None:
        result.matches_cubic_prototype = bool(matcher.fit_anonymous(structure, cubic_prototype()))
    return result


def make_matcher(sm_cfg: dict[str, Any]) -> StructureMatcher:
    """StructureMatcher con las tolerancias de configs/data.yaml."""
    return StructureMatcher(**sm_cfg)


# ---------------------------------------------------------------------------
# Ejecución sobre todas las estructuras descargadas en F1.1
# ---------------------------------------------------------------------------
def _identify_one(material_id: str, structure_dict: dict, cfg: dict[str, Any]) -> dict[str, Any]:
    """Envoltorio para joblib: un fallo en una estructura no detiene la corrida."""
    try:
        res = identify(
            Structure.from_dict(structure_dict), cfg, matcher=make_matcher(cfg["structure_matcher"])
        )
    except Exception as exc:  # se registra como clase `failed` con el mensaje
        res = Identification("failed", reasons=[f"{type(exc).__name__}: {exc}"[:200]])
    return {"material_id": material_id, **res.as_row()}


def identify_all(summary_path: Path, cfg: dict[str, Any], n_jobs: int = 1) -> pd.DataFrame:
    """Identifica todas las estructuras de `mp_abo3_summary.json.gz` (paralelo con loky)."""
    from joblib import Parallel, delayed

    docs = json.loads(gzip.decompress(summary_path.read_bytes()))
    rows = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_identify_one)(str(d["material_id"]), d["structure"], cfg) for d in docs
    )
    return pd.DataFrame(rows).sort_values("material_id", ignore_index=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="F1.3: identificación de perovskitas.")
    parser.add_argument("--config", type=Path, default=CONFIGS_DIR / "data.yaml")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--out", type=Path, default=INTERIM_DIR / "f1_3_identification.parquet")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text("utf-8"))["identification"]
    table = identify_all(RAW_DIR / "mp_abo3_summary.json.gz", cfg, n_jobs=args.n_jobs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out, index=False)
    print(table["cls"].value_counts().to_string())


if __name__ == "__main__":  # requisito en Windows (regla 2.2)
    main()
