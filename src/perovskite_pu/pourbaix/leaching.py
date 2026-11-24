"""F5.3: lixiviación de los sitios A y B según el diagrama de Pourbaix (tesis, OE4).

A partir de las especies estables en la ventana de potencial que guarda F5.1
(`stable_in_window`: combinaciones separadas por " | ", y en cada una especies separadas por
" + "), se clasifica cada especie como sólida (nombre terminado en "(s)") o disuelta (iones e
"(aq)"), y se identifican sus metales. Un sitio "se lixivia" si su metal aparece en alguna
especie disuelta estable en algún potencial de la ventana.

Es una lectura termodinámica del diagrama (qué fases son estables en equilibrio), no una
cinética de disolución. Descriptivo y exploratorio (solo tesis).
"""

from __future__ import annotations

from functools import cache

NON_METALS = {"O", "H"}


@cache
def species_metals(name: str) -> tuple[bool, frozenset[str]]:
    """(¿es sólida?, metales) de una especie con nombre de pymatgen.

    Ejemplos: "TiO2(s)" → (True, {Ti}); "Sr[+2]" → (False, {Sr});
    "Ti(OH)4(aq)" → (False, {Ti}).
    """
    from pymatgen.core import Composition
    from pymatgen.core.ion import Ion

    name = name.strip()
    if name.endswith("(s)"):
        comp = Composition(name[:-3])
        return True, frozenset(str(e) for e in comp.elements) - NON_METALS
    comp = Ion.from_formula(name).composition
    return False, frozenset(str(e) for e in comp.elements) - NON_METALS


def leaching_row(stable_in_window: str, a_el: str, b_el: str) -> dict[str, object]:
    """Qué metales pasan a disolución en la ventana y con qué especies.

    Devuelve si A y B aparecen disueltos, las especies disueltas de cada uno y los productos
    sólidos (ordenados, separados por "; ").
    """
    names = {s.strip() for combo in str(stable_in_window).split(" | ")
             for s in combo.split(" + ") if s.strip()}  # fmt: skip
    a_ions, b_ions, solids = set(), set(), set()
    for n in names:
        solid, metals = species_metals(n)
        if solid:
            solids.add(n)
            continue
        if a_el in metals:
            a_ions.add(n)
        if b_el in metals:
            b_ions.add(n)
    return {"A_dissolved": bool(a_ions), "B_dissolved": bool(b_ions),
            "A_species": "; ".join(sorted(a_ions)), "B_species": "; ".join(sorted(b_ions)),
            "solid_products": "; ".join(sorted(solids))}  # fmt: skip


def leaching_pattern(a_dissolved: bool, b_dissolved: bool) -> str:
    """Patrón de lixiviación: solo A, solo B, ambos o ninguno."""
    return {(True, False): "only_A", (False, True): "only_B",
            (True, True): "both", (False, False): "none"}[(a_dissolved, b_dissolved)]  # fmt: skip
