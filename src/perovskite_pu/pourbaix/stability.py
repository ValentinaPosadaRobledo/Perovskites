"""F5.1: ΔG_pbx continuo de la perovskita de cada composición (tesis, OE4 exploratorio).

Para cada composición ABO₃ de `train_pool` con polimorfo perovskita GGA/GGA+U (las de
`post_dft`), con las entradas de su sistema A–B–O–H descargadas por
`perovskite_pu.pourbaix.download`:

1. Se busca su entrada perovskita (`material_id_gga`) entre las sólidas de Pourbaix. MP usa
   ahora identificadores alfabéticos (`mp-aaaaagwx`) y las entradas de Pourbaix los numéricos
   (`mp-4651`); se convierten con `emmet.core.mpid.AlphaID`. Sin entrada: la composición queda
   fuera ("entrada GGA/GGA+U consistente", ROADMAP F5.1).
2. Diagrama de Pourbaix multielemento (A : B = 1 : 1, `filter_solids=True`) con cada
   concentración de iones.
3. ΔG_pbx (eV/átomo, `get_decomposition_energy`) en una rejilla de potenciales de la ventana,
   a cada pH: máximo (peor caso, criterio conservador), mínimo (mejor caso) y los potenciales
   donde ocurren, más la fase estable en el potencial del máximo.

ΔG_pbx ≤ ~0,5 eV/átomo (Singh et al., 2017) se usa solo como referencia. Determinista.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


def legacy_id(material_id: str) -> str:
    """Identificador numérico de MP (`mp-4651`) a partir del alfabético o del numérico."""
    from emmet.core.mpid import AlphaID

    s = str(material_id)
    suffix = s.split("-", 1)[1] if "-" in s else s
    return s if suffix.isdigit() else f"mp-{int(AlphaID(s))}"


def potential_grid(window: Sequence[float], step: float) -> np.ndarray:
    """Potenciales de la ventana (extremos incluidos), redondeados para evitar 0,15000000001."""
    lo, hi = float(window[0]), float(window[1])
    n = round((hi - lo) / step) + 1
    return np.round(np.linspace(lo, hi, n), 6)


def load_entries(path: Path) -> list[Any]:
    """Entradas de Pourbaix guardadas por la descarga (`as_dict` de cada una)."""
    from pymatgen.analysis.pourbaix_diagram import PourbaixEntry

    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [PourbaixEntry.from_dict(d) for d in json.load(fh)]


def find_target(entries: Sequence[Any], material_id: str) -> Any | None:
    """Entrada sólida de Pourbaix del `material_id` (alfabético o numérico), o None."""
    want = legacy_id(material_id)
    hits = [e for e in entries if e.phase_type == "Solid" and entry_mpid(e) == want]
    return hits[0] if hits else None


def entry_mpid(entry: Any) -> str:
    """`mp-<número>` de una entrada de Pourbaix, sea cual sea la forma de su `entry_id`.

    Según cómo se cree o se lea, `entry_id` es un objeto con `identifier`, un diccionario
    (tras `from_dict`) o un texto, con o sin sufijo de cálculo (`-GGA`), y el identificador
    puede ser numérico o alfabético. Siempre se normaliza al numérico (:func:`legacy_id`).
    """
    ident = entry.entry_id
    if isinstance(ident, dict):
        ident = ident.get("identifier", "")
    ident = getattr(ident, "identifier", ident)
    parts = str(ident).split("-")
    if len(parts) < 2 or parts[0] != "mp":
        return str(ident)  # p. ej. iones (`ion-0`): nunca coinciden con una perovskita
    return legacy_id("-".join(parts[:2]))


def window_extrema(diagram: Any, entry: Any, pH: float, potentials: np.ndarray) -> dict:
    """Máximo y mínimo de ΔG_pbx en la ventana, con sus potenciales (empate: el primero)."""
    dg = np.array([diagram.get_decomposition_energy(entry, pH, float(v)) for v in potentials])
    # Sin redondear, una ΔG_pbx plana elige su potencial por el ruido de coma flotante (1e-15).
    dg = np.round(dg, 9)
    i_max, i_min = int(np.argmax(dg)), int(np.argmin(dg))
    return {"dg_max": float(dg[i_max]), "v_at_max": float(potentials[i_max]),
            "dg_min": float(dg[i_min]), "v_at_min": float(potentials[i_min])}  # fmt: skip


def composition_rows(
    entries: Sequence[Any], target: Any, a_el: str, b_el: str, cond: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Filas (pH × concentración) de ΔG_pbx de una composición."""
    from pymatgen.analysis.pourbaix_diagram import PourbaixDiagram

    ratio = cond["comp_dict"]
    pots = potential_grid(cond["potential_window_V_SHE"], float(cond["potential_step_V"]))
    rows = []
    for conc in cond["ion_concentration_M"]:
        diagram = PourbaixDiagram(list(entries), comp_dict={a_el: ratio["A"], b_el: ratio["B"]},
                                  conc_dict={a_el: float(conc), b_el: float(conc)},
                                  filter_solids=bool(cond["filter_solids"]))  # fmt: skip
        for pH in cond["pH"]:
            ext = window_extrema(diagram, target, float(pH), pots)
            stable = diagram.get_stable_entry(float(pH), ext["v_at_max"]).name
            rows.append({"pH": float(pH), "ion_conc_M": float(conc), **ext,
                         "stable_at_max": stable})  # fmt: skip
    return rows
