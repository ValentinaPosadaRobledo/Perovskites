"""Tests de F5.3 (`perovskite_pu.pourbaix.leaching`), sin datos."""

from __future__ import annotations

from perovskite_pu.pourbaix.leaching import leaching_pattern, leaching_row, species_metals


def test_species_metals_solid_ion_and_aq():
    """Sólidos por el sufijo "(s)"; iones y "(aq)" disueltos; O y H no son metales."""
    assert species_metals("TiO2(s)") == (True, frozenset({"Ti"}))
    assert species_metals("Sr[+2]") == (False, frozenset({"Sr"}))
    assert species_metals("Ti(OH)4(aq)") == (False, frozenset({"Ti"}))
    assert species_metals("VO3[-1]") == (False, frozenset({"V"}))


def test_leaching_row_across_window():
    """Un metal se lixivia si está disuelto en alguna combinación de la ventana."""
    r = leaching_row("Sr[+2] + TiO2(s) | Sr[+2] + Ti(OH)4(aq)", "Sr", "Ti")
    assert r["A_dissolved"] and r["B_dissolved"]
    assert r["A_species"] == "Sr[+2]" and r["B_species"] == "Ti(OH)4(aq)"
    assert r["solid_products"] == "TiO2(s)"
    r = leaching_row("LaAlO3(s)", "La", "Al")
    assert not r["A_dissolved"] and not r["B_dissolved"]
    assert leaching_pattern(True, False) == "only_A" and leaching_pattern(False, False) == "none"
