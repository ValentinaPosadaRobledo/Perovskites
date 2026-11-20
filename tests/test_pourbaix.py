"""Tests de F5.1 (`perovskite_pu.pourbaix`), sin red ni datos descargados."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from perovskite_pu.pourbaix.download import candidate_systems
from perovskite_pu.pourbaix.run import summarize
from perovskite_pu.pourbaix.stability import (
    entry_mpid,
    legacy_id,
    potential_grid,
    window_extrema,
)


def test_legacy_id_alpha_and_numeric():
    """`mp-aaaaagwx` (alfabético) es `mp-4651`; los numéricos no cambian."""
    assert legacy_id("mp-aaaaagwx") == "mp-4651"
    assert legacy_id("mp-4651") == "mp-4651"


def test_potential_grid_includes_ends():
    """La ventana −0,2 a +0,4 V con paso 0,05 tiene 13 potenciales, extremos incluidos."""
    g = potential_grid([-0.2, 0.4], 0.05)
    assert len(g) == 13 and g[0] == -0.2 and g[-1] == 0.4 and 0.15 in g


class _FakeDiagram:
    """Diagrama de juguete: ΔG_pbx lineal en el potencial y en el pH."""

    def get_decomposition_energy(self, entry, pH, V):
        return 0.1 + 0.5 * V - 0.01 * pH


def test_window_extrema_max_min_and_potentials():
    """El peor caso está en el extremo superior de la ventana y el mejor en el inferior."""
    out = window_extrema(_FakeDiagram(), None, 7.4, potential_grid([-0.2, 0.4], 0.05))
    assert out["v_at_max"] == 0.4 and out["v_at_min"] == -0.2
    assert out["dg_max"] == pytest.approx(0.1 + 0.2 - 0.074)
    assert out["dg_min"] == pytest.approx(0.1 - 0.1 - 0.074)


def test_candidate_systems_only_train_pool_and_eligible():
    """Solo composiciones de `train_pool` elegibles en post_dft (nunca las de fuera)."""
    pool = pd.DataFrame({"composition": ["SrTiO3", "CaTiO3", "KNbO3"], "A": ["Sr", "Ca", "K"],
                         "B": ["Ti", "Ti", "Nb"], "label": [1, 0, 1]})  # fmt: skip
    meta = pd.DataFrame({"eligible_post_dft": [True, False, True, True],
                         "material_id_gga": ["mp-1", "mp-2", "mp-3", "mp-9"]},
                        index=["SrTiO3", "CaTiO3", "KNbO3", "BaZrO3"])  # fmt: skip
    out = candidate_systems(pool, meta)
    assert out["composition"].tolist() == ["KNbO3", "SrTiO3"]
    assert out["system"].tolist() == ["K-Nb", "Sr-Ti"]
    assert "BaZrO3" not in set(out["composition"])


def test_summarize_fraction_below_reference():
    """Fracción con ΔG_pbx máximo ≤ referencia, por etiqueta, pH y concentración."""
    t = pd.DataFrame({"label": [1, 1, 0, 0], "pH": 7.4, "ion_conc_M": 1e-6,
                      "dg_max": [0.2, 0.6, 0.4, 0.9], "dg_min": 0.0})  # fmt: skip
    s = summarize(t, 0.5).set_index("label")
    assert s.loc[1, "frac_dg_max_below_ref"] == 0.5 and s.loc[0, "n"] == 2
    assert np.isclose(s.loc[0, "dg_max_median"], 0.65)


class _Entry:
    """Entrada mínima con `entry_id` (objeto o texto), como tras `from_dict`."""

    def __init__(self, entry_id):
        self.entry_id = entry_id


def test_entry_mpid_object_and_text():
    """Acepta el identificador como objeto con `identifier` o como texto con sufijo."""

    class _Id:
        identifier = "mp-4651"

    assert entry_mpid(_Entry(_Id())) == "mp-4651"
    assert entry_mpid(_Entry("mp-4651-GGA+U")) == "mp-4651"
    assert entry_mpid(_Entry({"identifier": "mp-aaaaagwx", "suffix": "GGA"})) == "mp-4651"
    assert entry_mpid(_Entry("ion-0")) == "ion-0"
