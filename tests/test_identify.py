"""Tests de F1.3: identificación de perovskitas por conectividad.

Usan estructuras reales de Materials Project (CC BY 4.0) con clase conocida por la química,
guardadas en `tests/fixtures/mp_identification_cases.json`.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core import Structure

from perovskite_pu.data.identify import cubic_prototype, identify, octahedral_connectivity
from perovskite_pu.utils.paths import CONFIGS_DIR

CASES = json.loads(
    (Path(__file__).parent / "fixtures" / "mp_identification_cases.json").read_text()
)["cases"]


@pytest.fixture(scope="module")
def id_cfg() -> dict:
    return yaml.safe_load((CONFIGS_DIR / "data.yaml").read_text("utf-8"))["identification"]


def _case(formula: str, spacegroup: str) -> dict:
    return next(c for c in CASES if c["formula"] == formula and c["spacegroup"] == spacegroup)


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c['formula']}-{c['spacegroup']}")
def test_known_structures(case, id_cfg):
    """Cada estructura de referencia recibe la clase (y el sitio B) que dicta la química."""
    res = identify(Structure.from_dict(case["structure"]), id_cfg)
    assert res.cls == case["expected_cls"], f"{case['why']}: {res.reasons}"
    if case["expected_b"] is not None:
        assert res.b_element == case["expected_b"]


def test_angle_threshold_matters_for_tilted_pnma(id_cfg):
    """GdScO3 (ángulo mínimo ~139°) es perovskita con 135° y caso límite con 140° (sensibilidad)."""
    s = Structure.from_dict(_case("GdScO3", "Pnma")["structure"])
    strict = copy.deepcopy(id_cfg)
    strict["octahedral_connectivity"]["min_b_o_b_angle_deg"] = strict["octahedral_connectivity"][
        "sensitivity_min_b_o_b_angle_deg"
    ]
    assert identify(s, id_cfg).cls == "perovskite"
    assert identify(s, strict).cls == "perovskite_borderline"


def test_lone_pair_rule_matters_for_bifeo3(id_cfg):
    """BiFeO3 R3c es perovskita gracias a la regla del par libre; sin ella, caso límite."""
    s = Structure.from_dict(_case("FeBiO3", "R3c")["structure"])
    no_lp = copy.deepcopy(id_cfg)
    no_lp["coordination"].pop("lone_pair_a_rule")
    assert identify(s, id_cfg).cls == "perovskite"
    res = identify(s, no_lp)
    assert res.cls == "perovskite_borderline" and res.b_element == "Fe"


def test_lone_pair_rule_does_not_rescue_linbo3(id_cfg):
    """Li no es catión con par libre: LiNbO3 sigue fuera aunque el ángulo esté cerca del umbral."""
    res = identify(Structure.from_dict(_case("LiNbO3", "R3c")["structure"]), id_cfg)
    assert res.cls == "perovskite_borderline"
    assert res.a_cn_max == 6


def test_cubic_connectivity_counts():
    """Perovskita cúbica de 5 átomos: 3 enlaces B–B por vértice, sin aristas ni caras, 180°."""
    s = cubic_prototype()
    s.add_oxidation_state_by_element({"Sr": 2, "Ti": 4, "O": -2})
    nn = {i: CrystalNN(cation_anion=True).get_nn_info(s, i) for i in (0, 1)}
    c = octahedral_connectivity(s, nn, [1])
    assert (c["n_corner_links"], c["n_edge_links"], c["n_face_links"]) == (3, 0, 0)
    assert c["max_shared_o"] == 1 and c["o_with_2b_frac"] == pytest.approx(1.0)
    assert c["bob_angle_min"] == pytest.approx(180.0)
