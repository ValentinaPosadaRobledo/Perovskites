"""Tests de la estimación de la prior de clase (F3.2), con datos sintéticos de c conocida."""

from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from perovskite_pu.evaluation.prior import (
    elkan_noto_label_frequency,
    precision_under_scar,
    prior_from_label_frequency,
    tice_delta,
    tice_discretize,
    tice_label_frequency,
    tice_lower_bound,
)


def scar_data(n_pos=600, n_neg=1400, c=0.3, d=4, shift=3.0, seed=0):
    """Datos PU bajo SCAR: cada positiva se etiqueta con probabilidad c, al azar.

    Las clases se solapan solo en parte, así que existe una subregión con solo positivas
    (supuesto de subdominio positivo de TIcE).
    """
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(shift / 2, 1.0, (n_pos, d)), rng.normal(-shift / 2, 1.0, (n_neg, d))])
    y = np.r_[np.ones(n_pos, int), np.zeros(n_neg, int)]
    s = ((y == 1) & (rng.random(y.size) < c)).astype(int)
    return X, y, s


def test_prior_from_label_frequency_algebra():
    est = prior_from_label_frequency(0.5, n_labeled=100, n_unlabeled=900)
    # α = n_L / (c N) = 100 / 500 = 0,2; positivas ocultas = n_L (1 − c) / c = 100.
    assert est.alpha == pytest.approx(0.2)
    assert est.hidden_positives == pytest.approx(100.0)
    assert est.pi_u == pytest.approx(100 / 900)
    assert not est.clipped


def test_prior_clipping_is_flagged():
    low = prior_from_label_frequency(0.05, 100, 900)  # c < n_L/N = 0,1 → α > 1
    assert low.clipped and low.alpha == pytest.approx(1.0) and low.pi_u == pytest.approx(1.0)
    high = prior_from_label_frequency(1.3, 100, 900)
    assert high.clipped and high.pi_u == pytest.approx(0.0)
    with pytest.raises(ValueError):
        prior_from_label_frequency(0.5, 0, 10)


def test_lower_bound_is_conservative_and_monotone():
    # Con más filas, o más confianza tolerada (δ mayor), la corrección es menor.
    assert tice_lower_bound(30, 100, 0.1) < 0.3
    assert tice_lower_bound(300, 1000, 0.1) > tice_lower_bound(30, 100, 0.1)
    assert tice_lower_bound(30, 100, 0.2) > tice_lower_bound(30, 100, 0.05)
    assert tice_lower_bound(0, 0, 0.1) == -math.inf
    # Con ĉ lejos de 0,5 la cota es más ajustada (Ec. 6 frente a Ec. 7).
    assert tice_lower_bound(30, 100, 0.1, c_prior=0.2) > tice_lower_bound(30, 100, 0.1)


def test_tice_delta_matches_equation_8():
    assert tice_delta(10_000_000) == pytest.approx(0.025)  # suelo de 0,025
    assert tice_delta(250) == pytest.approx(1 / (1 + 0.004 * 250))  # = 0,5


def test_discretize_uniform_and_quantile():
    x = np.r_[np.arange(10.0), 1000.0][:, None]  # un valor extremo
    uni = tice_discretize(x, n_bins=4, binning="uniform")[:, 0]
    assert set(uni[:-1]) == {0} and uni[-1] == 3  # el extremo deja casi todo en un intervalo
    qua = tice_discretize(x, n_bins=4, binning="quantile")[:, 0]
    assert len(set(qua)) == 4  # por cuantiles, intervalos de tamaño parecido
    assert (tice_discretize(np.ones((5, 1)))[:, 0] == 0).all()  # constante: un intervalo
    with pytest.raises(ValueError):
        tice_discretize(x, binning="otro")


@pytest.mark.parametrize("c_true", [0.2, 0.5])
def test_elkan_noto_recovers_c_under_scar(c_true):
    X, _, s = scar_data(c=c_true, seed=1)
    model = CalibratedClassifierCV(LogisticRegression(max_iter=1000), method="sigmoid", cv=3)
    c_hat, oof = elkan_noto_label_frequency(X, s, model, n_splits=5, random_state=0)
    assert np.isfinite(oof).all()
    assert c_hat == pytest.approx(c_true, abs=0.08)


@pytest.mark.parametrize("c_true", [0.5, 0.8])  # régimen de train_pool (c ≥ 0,21)
def test_tice_recovers_c_under_scar(c_true):
    # Con N = 2000, TIcE varía ±0,1 entre conjuntos de datos: se exige que la media de 5
    # conjuntos sea cercana a c y que ninguno se aleje mucho.
    estimates = []
    for seed in range(5):
        X, _, s = scar_data(c=c_true, seed=seed)
        c_hat, detail = tice_label_frequency(X, s, n_folds=5, random_state=seed)
        assert len(detail["first_pass"]) == len(detail["second_pass"]) == 5
        estimates.append(c_hat)
    assert np.mean(estimates) == pytest.approx(c_true, abs=0.05)
    assert np.max(np.abs(np.array(estimates) - c_true)) < 0.12


def test_tice_is_deterministic_and_rejects_missing_values():
    X, _, s = scar_data(seed=3)
    a = tice_label_frequency(X, s, random_state=7)
    b = tice_label_frequency(X, s, random_state=7)
    assert a == b
    X[0, 0] = np.nan
    with pytest.raises(ValueError):
        tice_label_frequency(X, s)


def test_tice_does_not_see_labels_through_features():
    # Si una feature fuera la propia etiqueta, la región "feature = 1" tendría solo
    # etiquetadas y TIcE daría c ≈ 1: el test documenta que la auditoría de features
    # (regla 2.1.8) es lo que protege de esto, no el estimador.
    X, _, s = scar_data(seed=4)
    leaky = np.column_stack([X, s])
    c_hat, _ = tice_label_frequency(leaky, s, random_state=0)
    assert c_hat > 0.8


def test_precision_under_scar():
    # Sin positivas ocultas (π_U = 0), las U predichas son todas falsos positivos.
    out = precision_under_scar(tpr=0.9, ppr_u=0.1, pi_u=0.0, n_pos=100, n_u=1000)
    assert out["precision_u_scar"] == pytest.approx(0.0)
    assert out["precision_scar"] == pytest.approx(90 / (90 + 100))
    # Si todas las U predichas fueran positivas ocultas, la precisión entre U sería 1.
    full = precision_under_scar(tpr=0.5, ppr_u=0.05, pi_u=0.1, n_pos=100, n_u=1000)
    assert full["precision_u_scar"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Ejecución del experimento (piezas, sin datos reales)
# ---------------------------------------------------------------------------
def test_population_mask_keeps_all_positives():
    import pandas as pd

    from perovskite_pu.evaluation.prior_run import population_mask

    pool = pd.DataFrame({"label": [1, 1, 0, 0, 0], "flag": [True, False, True, False, False]})
    assert population_mask(pool, {"exclude_unlabeled_flag": None}).all()
    keep = population_mask(pool, {"exclude_unlabeled_flag": "flag"})
    assert keep.tolist() == [True, True, False, True, True]  # una P con bandera se queda


def test_estimate_one_is_deterministic_and_handles_missing_values():
    from perovskite_pu.evaluation.prior_run import estimate_one

    X, _, s = scar_data(c=0.5, seed=5)
    X[::7, 0] = np.nan  # huecos: los imputa el Pipeline (EN) o la imputación previa (TIcE)
    en = {"kind": "elkan_noto", "role": "principal", "n_splits": 3, "learner": "random_forest",
          "params": {"n_estimators": 30, "min_samples_leaf": 5},
          "calibration": {"method": "isotonic", "cv": 3}}  # fmt: skip
    tice = {"kind": "tice", "role": "principal", "n_folds": 5, "k": 5, "max_splits": 100,
            "n_bins": 4, "binning": "uniform"}  # fmt: skip
    prep = {"add_missing_indicator": True}
    for spec in (en, tice):
        a, _ = estimate_one(X, s, "m", spec, prep, seed=0)
        b, _ = estimate_one(X, s, "m", spec, prep, seed=0)
        c, _ = estimate_one(X, s, "m", spec, prep, seed=1)
        assert a == b  # misma semilla, mismo resultado
        assert a["c"] != c["c"]  # otra semilla, otros folds
        assert 0 <= a["pi_u"] <= 1 and a["n_labeled"] == int(s.sum())
    with pytest.raises(ValueError):
        estimate_one(X, s, "m", {"kind": "otro"}, prep, seed=0)


def test_summarize_prior_one_row_per_method_and_population():
    import pandas as pd

    from perovskite_pu.evaluation.prior_run import summarize_prior

    grid = [(p, m, sd) for p in ("all", "sub") for m in ("a", "b") for sd in range(3)]
    raw = pd.DataFrame([{"population": p, "method": m, "kind": "tice", "role": "principal",
                         "seed": sd, "c": 0.5 + 0.01 * sd, "alpha": 0.3, "pi_u": 0.2,
                         "hidden_positives": 10.0, "clipped": False, "n_labeled": 5,
                         "n_unlabeled": 50} for p, m, sd in grid])  # fmt: skip
    out = summarize_prior(raw)
    assert len(out) == 4 and (out["n_seeds"] == 3).all()
    assert out["c_mean"].iloc[0] == pytest.approx(0.51)
    assert (out["c_ci95_low"] < out["c_mean"]).all()


def test_prior_config_is_complete():
    import yaml

    from perovskite_pu.utils.paths import CONFIGS_DIR

    cfg = yaml.safe_load((CONFIGS_DIR / "experiments" / "f3_2_prior.yaml").read_text("utf-8"))
    kinds = {spec["kind"] for spec in cfg["methods"].values()}
    assert kinds == {"elkan_noto", "tice"}  # ≥ 2 métodos distintos (preregistro §7)
    roles = [spec["role"] for spec in cfg["methods"].values()]
    assert roles.count("principal") == 2
    assert "all" in cfg["populations"]
    assert "seed" not in yaml.safe_dump(cfg).replace("seeds.yaml", "")  # semillas solo en seeds
