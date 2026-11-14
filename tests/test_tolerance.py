"""Tests de la reimplementación de t, μ y τ de Bartel et al. (2019) (F2.1).

La validación principal reproduce los 576 compuestos de su TableS1 (copia en
`tests/fixtures/bartel2019_TableS1.csv`, licencia MIT). TableS1 redondea a 2 decimales, así que
la tolerancia es de media centésima.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from perovskite_pu.features.tolerance import TAU_THRESHOLD, default_tables, tau_descriptors

FIXTURES_DIR = Path(__file__).parent / "fixtures"
TOL = 0.005 + 1e-9  # error máximo por el redondeo a 2 decimales de TableS1


@pytest.fixture(scope="module")
def tables():
    """Radios de Shannon y electronegatividades incluidos en el paquete."""
    return default_tables()


@pytest.fixture(scope="module")
def table_s1(tables) -> pd.DataFrame:
    """TableS1 de Bartel con nuestras predicciones al lado (columnas `ours_*`)."""
    shannon, chi = tables
    t = pd.read_csv(FIXTURES_DIR / "bartel2019_TableS1.csv")
    ours = pd.DataFrame([tau_descriptors(f, shannon, chi).as_row() for f in t["ABX3"]])
    return pd.concat([t, ours.add_prefix("ours_")], axis=1)


def test_table_s1_has_576_compounds(table_s1):
    assert len(table_s1) == 576


def test_site_and_oxidation_assignment_match_table_s1(table_s1):
    # A, B y los estados de oxidación deben coincidir en TODOS los compuestos.
    for col in ("A", "B", "nA", "nB"):
        assert (table_s1[f"ours_{col}"] == table_s1[col]).all(), col


def test_radii_match_table_s1(table_s1):
    for ours, ref in (("rA", "rA (Ang)"), ("rB", "rB (Ang)"), ("rX", "rX (Ang)")):
        assert ((table_s1[f"ours_{ours}"] - table_s1[ref]).abs() <= TOL).all(), ours


def test_t_and_tau_match_table_s1(table_s1):
    assert ((table_s1["ours_t"] - table_s1["t"]).abs() <= TOL).all()
    assert ((table_s1["ours_tau"] - table_s1["tau"]).abs() <= TOL).all()


def test_tau_prediction_matches_table_s1(table_s1):
    # tau_pred = 1 si τ < 4,18 y −1 en caso contrario.
    ours = (table_s1["ours_tau"] < TAU_THRESHOLD).map({True: 1, False: -1})
    assert (ours == table_s1["tau_pred"]).all()


def test_srtio3_by_hand(tables):
    # SrTiO3: Sr2+ (NC 12, 1,44 Å), Ti4+ (NC 6, 0,605 Å), O2− (NC 6, 1,40 Å).
    res = tau_descriptors("SrTiO3", *tables)
    assert (res.A, res.B, res.nA, res.nB) == ("Sr", "Ti", 2, 4)
    assert (res.A_cn, res.B_cn) == (12, 6) and not res.notes
    rA, rB, rX = res.rA, res.rB, res.rX
    assert res.t == pytest.approx((rA + rX) / (math.sqrt(2) * (rB + rX)))
    assert res.mu == pytest.approx(rB / rX)
    q = rA / rB
    assert res.tau == pytest.approx(rX / rB - 2 * (2 - q / math.log(q)))
    assert res.tau < TAU_THRESHOLD


def test_order_of_formula_does_not_matter(tables):
    a, b = tau_descriptors("SrTiO3", *tables), tau_descriptors("TiSrO3", *tables)
    assert (a.A, a.B, a.tau) == (b.A, b.B, b.tau)


def test_no_charge_balance_gives_nan_with_note(tables):
    # Na (1+ fijo) y K (1+ fijo) no pueden equilibrar O3 (6−).
    res = tau_descriptors("NaKO3", *tables)
    assert math.isnan(res.tau) and math.isnan(res.t) and math.isnan(res.mu)
    assert any("equilibre la carga" in n for n in res.notes)


def test_non_abx3_formula_is_rejected_with_note(tables):
    res = tau_descriptors("Sr2TiO4", *tables)
    assert math.isnan(res.tau) and res.notes and "No es ABX3" in res.notes[0]


def test_substitute_radius_is_flagged(tables):
    # Bi3+ no tiene radio de Shannon en NC 12: se usa el NC disponible más cercano.
    res = tau_descriptors("BiFeO3", *tables)
    assert res.A == "Bi" and res.A_cn != 12
    assert any("r_A sustituto" in n for n in res.notes)


def test_vendored_tables_are_unmodified():
    # Las copias de Bartel deben ser idénticas al original: su SHA-256 está en las notas de
    # licencia. Protege contra normalizaciones de finales de línea (p. ej. en Windows).
    import hashlib
    import re

    pkg_data = Path(__file__).parents[1] / "src" / "perovskite_pu" / "features" / "data"
    notes = {
        pkg_data / "LICENSE_bartel2019.txt": pkg_data,
        FIXTURES_DIR / "bartel2019_TableS1.LICENSE.txt": FIXTURES_DIR,
    }
    checked = 0
    for note, folder in notes.items():
        text = note.read_text(encoding="utf-8")
        for name, digest in re.findall(r"^\s+(\S+)\s+sha256 ([0-9a-f]{64})$", text, re.M):
            assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest, name
            checked += 1
        # La nota de la fixture da solo el hash, sin nombre: el archivo es la TableS1.
        for digest in re.findall(r"^\s+sha256 ([0-9a-f]{64})$", text, re.M):
            table = FIXTURES_DIR / "bartel2019_TableS1.csv"
            assert hashlib.sha256(table.read_bytes()).hexdigest() == digest
            checked += 1
    assert checked == 3
