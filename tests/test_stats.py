"""Tests de las pruebas estadísticas preregistradas (F4.2, `evaluation.stats`).

Todo con datos sintéticos: no se lee ningún archivo de datos ni resultados.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from perovskite_pu.evaluation.metrics import expected_hits_at_ppr, hit_probability_at_ppr
from perovskite_pu.evaluation.stats import (
    BootstrapResult,
    bootstrap_rng,
    bootstrap_units,
    composition_hits,
    h1_decision,
    h2_decision,
    h3_decision,
    hierarchical_bootstrap,
    hierarchical_bootstrap_distribution,
    holm,
    matched_ppr_hits,
    nadeau_bengio,
)

# ---------------------------------------------------------------------------
# Aciertos a P(ŷ = 1) fija, por fila
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", [0.0, 0.1, 0.25, 0.37, 0.5, 1.0])
def test_hit_probability_sums_to_expected_hits(target):
    """Sumada sobre los positivos, coincide con `expected_hits_at_ppr` (con empates)."""
    rng = np.random.default_rng(0)
    scores = rng.integers(0, 6, size=40).astype(float)  # muchos empates
    s = rng.integers(0, 2, size=40)
    prob = hit_probability_at_ppr(scores, target)
    assert prob.sum() == pytest.approx(target * scores.size)
    assert prob[s == 1].sum() == pytest.approx(expected_hits_at_ppr(s, scores, target))
    assert ((prob >= 0) & (prob <= 1)).all()


def test_hit_probability_without_ties_is_binary():
    """Sin empates en el corte, cada fila entra o no entra."""
    prob = hit_probability_at_ppr([0.9, 0.8, 0.7, 0.6], 0.5)
    np.testing.assert_allclose(prob, [1, 1, 0, 0])


# ---------------------------------------------------------------------------
# h(c) por composición
# ---------------------------------------------------------------------------


def _raw(model: str, scheme: str, seeds: int = 3) -> pd.DataFrame:
    """Tabla tipo `raw/seed_n.csv`: 4 positivas y 4 U en 2 folds, en cada semilla."""
    rows = []
    rng = np.random.default_rng(1)
    comps = [f"P{i}" for i in range(4)] + [f"U{i}" for i in range(4)]
    for seed in range(seeds):
        for i, c in enumerate(comps):
            score = float(rng.random())
            rows.append({"baseline": model, "scheme": scheme, "seed": seed, "fold": i % 2,
                         "composition": c, "label": int(c.startswith("P")), "score": score,
                         "pred": int(score > 0.5)})  # fmt: skip
    return pd.DataFrame(rows)


def test_composition_hits_is_mean_over_seeds():
    """h(c) es la fracción de semillas con acierto, y su media es la TPR micro-agregada."""
    raw = _raw("m", "loeo_B")
    h = composition_hits(raw, "m", "loeo_B")
    pos = raw[raw["label"] == 1]
    assert set(h.index) == {"P0", "P1", "P2", "P3"}
    assert h.mean() == pytest.approx(pos["pred"].mean())


def test_composition_hits_rejects_duplicates():
    """Un positivo evaluado dos veces en la misma semilla es un error de partición."""
    raw = _raw("m", "loeo_B")
    dup = pd.concat([raw, raw[(raw["composition"] == "P0") & (raw["seed"] == 0)]])
    with pytest.raises(ValueError, match="más de una vez"):
        composition_hits(dup, "m", "loeo_B")


def test_matched_ppr_hits_matches_reference_ppr():
    """El modelo, umbralizado a la P(ŷ=1) de la referencia, predice esa misma fracción."""
    model, ref = _raw("pu", "family"), _raw("tau", "family")
    ref["pred"] = (ref["composition"].isin(["P0", "P1", "U0"])).astype(int)  # P(ŷ=1) = 3/8
    out = matched_ppr_hits(model, ref, "family")
    assert list(out.index) == ["P0", "P1", "P2", "P3"]
    np.testing.assert_allclose(out["h_ref"], [1, 1, 0, 0])
    # Media sobre composiciones = media sobre semillas de los aciertos esperados.
    expected = []
    for (_seed, _fold), g in model.groupby(["seed", "fold"]):
        r = ref[(ref["seed"] == _seed) & (ref["fold"] == _fold)]
        expected.append(expected_hits_at_ppr(g["label"], g["score"], r["pred"].mean()))
    assert out["h_model"].sum() * 3 == pytest.approx(sum(expected))


def test_matched_ppr_hits_exclude_and_mismatch():
    """`exclude` quita composiciones; folds distintos entre modelos fallan."""
    model, ref = _raw("pu", "family"), _raw("tau", "family")
    out = matched_ppr_hits(model, ref, "family", exclude={"P0"})
    assert "P0" not in out.index
    bad = model.copy()
    bad.loc[bad["composition"] == "P1", "fold"] = 1 - bad.loc[bad["composition"] == "P1", "fold"]
    with pytest.raises(ValueError, match="Folds distintos"):
        matched_ppr_hits(bad, ref, "family")


# ---------------------------------------------------------------------------
# Unidades y semillas del bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_units_by_scheme():
    """Elemento A en loeo_A, B en los demás; `family` estratifica por familia."""
    comps = pd.DataFrame({"A": ["Sr", "Ba", "Ca"], "B": ["Ti", "Zr", "Ti"]},
                         index=["SrTiO3", "BaZrO3", "CaTiO3"])  # fmt: skip
    units, strata = bootstrap_units(comps, "loeo_A")
    assert list(units) == ["Sr", "Ba", "Ca"] and strata is None
    units, strata = bootstrap_units(comps, "cluster")
    assert list(units) == ["Ti", "Zr", "Ti"] and strata is None
    units, strata = bootstrap_units(comps, "family", {"Ti": "3d", "Zr": "4d_5d"})
    assert list(strata) == ["3d", "4d_5d", "3d"]
    with pytest.raises(ValueError, match="sin familia"):
        bootstrap_units(comps, "family", {"Ti": "3d"})
    with pytest.raises(ValueError, match="no está definido"):
        bootstrap_units(comps, "random")


def test_bootstrap_rng_is_reproducible_and_distinct():
    """Mismo (hipótesis, esquema) → mismo flujo; distinto → otro flujo."""
    a = bootstrap_rng(20261113, "H1", "family").random(5)
    b = bootstrap_rng(20261113, "H1", "family").random(5)
    c = bootstrap_rng(20261113, "H2", "family").random(5)
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


# ---------------------------------------------------------------------------
# Bootstrap jerárquico
# ---------------------------------------------------------------------------


def _clustered(n_units: int = 30, per_unit: int = 8, effect: float = 0.1, sd_unit: float = 0.1,
               seed: int = 0) -> tuple[pd.Series, pd.Series]:  # fmt: skip
    """d(c) con un efecto medio, variación entre unidades y ruido dentro de cada unidad."""
    rng = np.random.default_rng(seed)
    unit_effect = effect + rng.normal(0, sd_unit, n_units)
    d, u = [], []
    for k in range(n_units):
        d += list(unit_effect[k] + rng.normal(0, 0.2, per_unit))
        u += [f"u{k}"] * per_unit
    idx = [f"c{i}" for i in range(len(d))]
    return pd.Series(d, index=idx), pd.Series(u, index=idx)


def test_bootstrap_estimate_ci_and_determinism():
    """La estimación es la media; el IC la contiene; misma semilla → mismo resultado."""
    d, u = _clustered()
    r1 = hierarchical_bootstrap(d, u, n_boot=2000, rng=np.random.default_rng(5))
    r2 = hierarchical_bootstrap(d, u, n_boot=2000, rng=np.random.default_rng(5))
    assert isinstance(r1, BootstrapResult)
    assert r1.estimate == pytest.approx(d.mean())
    assert r1.ci_low < r1.estimate < r1.ci_high
    assert r1 == r2
    assert r1.n_units == 30 and r1.n_compositions == 240


def test_bootstrap_ci_widens_with_between_unit_variation():
    """El nivel 1 recoge la variación entre unidades químicas (agrupamiento)."""
    d0, u0 = _clustered(sd_unit=0.0, seed=3)
    d1, u1 = _clustered(sd_unit=0.3, seed=3)
    w0 = hierarchical_bootstrap(d0, u0, n_boot=2000, rng=np.random.default_rng(0))
    w1 = hierarchical_bootstrap(d1, u1, n_boot=2000, rng=np.random.default_rng(0))
    assert (w1.ci_high - w1.ci_low) > 2 * (w0.ci_high - w0.ci_low)


def test_bootstrap_coverage_on_synthetic_clusters():
    """Cobertura aproximada del IC 95 % frente al efecto real (población de unidades)."""
    hits = 0
    n_rep = 60
    for rep in range(n_rep):
        d, u = _clustered(n_units=25, effect=0.05, sd_unit=0.15, seed=100 + rep)
        r = hierarchical_bootstrap(d, u, n_boot=1000, rng=np.random.default_rng(rep))
        hits += r.ci_low <= 0.05 <= r.ci_high
    assert hits / n_rep >= 0.85  # holgura por el número finito de repeticiones


def test_stratified_bootstrap_keeps_strata_sizes():
    """Con una unidad constante por estrato, el remuestreo estratificado no varía."""
    d = pd.Series([0.1, 0.1, 0.5, 0.5, 0.9], index=list("abcde"))
    units = pd.Series(["Ti", "Ti", "Zr", "Zr", "Ce"], index=list("abcde"))
    strata = pd.Series(["3d", "3d", "4d", "4d", "4f"], index=list("abcde"))
    dist = hierarchical_bootstrap_distribution(d.to_numpy(), units.to_numpy(),
                                               strata.to_numpy(), n_boot=200,
                                               rng=np.random.default_rng(0))  # fmt: skip
    np.testing.assert_allclose(dist, d.mean())
    # Sin estratos, el nivel 1 puede repetir unidades y la media sí varía.
    free = hierarchical_bootstrap_distribution(
        d.to_numpy(), units.to_numpy(), None, n_boot=200, rng=np.random.default_rng(0)
    )
    assert free.std() > 0


def test_bootstrap_p_values():
    """Todas las diferencias positivas → p bilateral 0; margen por debajo → p unilateral 0."""
    d = pd.Series(np.linspace(0.05, 0.15, 40), index=[f"c{i}" for i in range(40)])
    u = pd.Series([f"u{i % 8}" for i in range(40)], index=d.index)
    r = hierarchical_bootstrap(d, u, margin=-0.05, n_boot=500, rng=np.random.default_rng(0))
    assert r.p_two_sided == 0.0 and r.p_le_margin == 0.0 and r.p_le_zero == 0.0
    r0 = hierarchical_bootstrap(d - d.mean(), u, n_boot=2000, rng=np.random.default_rng(0))
    assert r0.p_two_sided > 0.5  # efecto nulo exacto: p grande


def test_bootstrap_input_checks():
    """Composiciones sin unidad, repetidas o NaN fallan."""
    d = pd.Series([0.1, 0.2], index=["a", "b"])
    with pytest.raises(ValueError, match="sin unidad"):
        hierarchical_bootstrap(d, pd.Series(["u"], index=["a"]))
    with pytest.raises(ValueError, match="repetidas"):
        hierarchical_bootstrap(pd.Series([0.1, 0.2], index=["a", "a"]),
                               pd.Series(["u", "u"], index=["a", "b"]))  # fmt: skip
    with pytest.raises(ValueError, match="NaN"):
        hierarchical_bootstrap_distribution([0.1, np.nan], ["u", "u"], n_boot=5)


# ---------------------------------------------------------------------------
# Holm y Nadeau–Bengio
# ---------------------------------------------------------------------------


def test_holm_known_values():
    """Ejemplo de referencia: ajuste paso descendente, monótono y acotado a 1."""
    p = [0.01, 0.04, 0.03, 0.005]
    np.testing.assert_allclose(holm(p), [0.03, 0.06, 0.06, 0.02])
    np.testing.assert_allclose(holm([0.5, 0.9]), [1.0, 1.0])
    with pytest.raises(ValueError):
        holm([0.1, np.nan])


def test_nadeau_bengio_formula_and_correction():
    """t = media / sqrt((1/J + n_test/n_train) s²); más conservador que el t simple."""
    rng = np.random.default_rng(0)
    diffs = 0.02 + rng.normal(0, 0.03, 50)
    out = nadeau_bengio(diffs, n_train=800, n_test=200)
    se = np.sqrt((1 / 50 + 200 / 800) * diffs.var(ddof=1))
    assert out["t"] == pytest.approx(diffs.mean() / se)
    assert out["df"] == 49
    naive_t = diffs.mean() / np.sqrt(diffs.var(ddof=1) / 50)
    assert abs(out["t"]) < abs(naive_t)
    assert out["ci_low"] < out["mean"] < out["ci_high"]


def test_nadeau_bengio_degenerate():
    """Sin variabilidad: p = 1 si la media es 0 y p = 0 si no."""
    assert nadeau_bengio([0.0, 0.0, 0.0], 80, 20)["p_two_sided"] == 1.0
    assert nadeau_bengio([0.1, 0.1, 0.1], 80, 20)["p_two_sided"] == 0.0


# ---------------------------------------------------------------------------
# Decisiones de H1–H3
# ---------------------------------------------------------------------------


def _res(estimate: float, p2: float, p_margin: float = 1.0, p_zero: float = 1.0,
         margin: float = 0.0) -> BootstrapResult:  # fmt: skip
    """Resultado de bootstrap fabricado para probar las reglas de decisión."""
    return BootstrapResult(estimate, estimate - 0.1, estimate + 0.1, p2, p_margin, p_zero,
                           margin, 10, 100, 10_000)  # fmt: skip


def test_h1_decision_requires_significance_and_relevance():
    """Criterio: p de Holm < 0,05 y ΔTPR ≥ δ₁; global = intersección-unión."""
    res = {"loeo_A": _res(0.044, 0.001), "loeo_B": _res(0.20, 0.0001),
           "family": _res(0.21, 0.0001), "cluster": _res(0.12, 0.02)}  # fmt: skip
    t = h1_decision(res)
    assert not t.loc["loeo_A", "criterion"]  # significativo pero < δ₁
    assert t.loc["family", "criterion"] and t.loc["loeo_B", "criterion"]
    assert t.loc["cluster", "p_holm"] == pytest.approx(0.02)  # el mayor de 4: ×1
    assert not t["global"].iloc[0]


def test_h2_decision_noninferiority_and_superiority():
    """No inferioridad con Holm sobre P*(Δ* ≤ −δ₂) < 0,025; superioridad aparte."""
    m = -0.05
    res = {"loeo_A": _res(0.02, 0.1, p_margin=0.0001, p_zero=0.001, margin=m),
           "loeo_B": _res(0.00, 0.9, p_margin=0.004, p_zero=0.5, margin=m),
           "family": _res(-0.056, 0.0, p_margin=0.6, p_zero=1.0, margin=m),
           "cluster": _res(-0.04, 0.0, p_margin=0.3, p_zero=1.0, margin=m)}  # fmt: skip
    t = h2_decision(res)
    assert t.loc["loeo_A", "noninferior"] and t.loc["loeo_A", "superior"]
    assert t.loc["loeo_B", "noninferior"] and not t.loc["loeo_B", "superior"]
    assert not t.loc["family", "noninferior"]
    assert not t["global_noninferior"].iloc[0]
    with pytest.raises(ValueError, match="margin"):
        h2_decision({"family": _res(0.0, 1.0)})


def test_h3_decision():
    """H3: D > 0 y p de Holm < α."""
    t = h3_decision({"family": _res(0.05, 0.001), "cluster": _res(-0.02, 0.001)})
    assert t.loc["family", "criterion"] and not t.loc["cluster", "criterion"]
