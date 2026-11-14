"""Tests del ejecutor de experimentos y de las líneas base (F2.4), con datos sintéticos."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from perovskite_pu.splits.schemes import Fold
from perovskite_pu.training.baselines import RandomScores, make_estimator, rule_scores
from perovskite_pu.training.runner import (
    evaluate_fold,
    frozen_folds,
    inner_threshold,
    seed_metrics,
    summarize,
    task_seed,
)

INNER = {"n_splits": 3, "target_recall": 0.90}


@pytest.fixture
def inputs() -> pd.DataFrame:
    """60 composiciones: las 20 primeras positivas; E_hull baja en las positivas."""
    rng = np.random.default_rng(0)
    n = 60
    label = np.r_[np.ones(20, int), np.zeros(40, int)]
    df = pd.DataFrame(
        {
            "tau": np.where(label == 1, 3.0, 6.0) + rng.normal(0, 0.5, n),
            "t": rng.uniform(0.75, 1.1, n),
            "ehull": np.where(label == 1, 0.01, 0.3) + rng.uniform(0, 0.05, n),
            "charge_balanced_icsd": np.r_[np.ones(30), np.zeros(30)],
            "spacegroup_number": np.where(label == 1, "62", "221"),
            "crystal_system": np.where(label == 1, "Orthorhombic", "Cubic"),
        },
        index=pd.Index([f"C{i:02d}O3" for i in range(n)], name="composition"),
    )
    df["all"] = True
    df["tau_defined"] = True
    df["t_defined"] = True
    df["post_dft"] = np.arange(n) % 10 != 9  # una de cada diez, no elegible
    df["label"] = label
    return df


@pytest.fixture
def fold() -> Fold:
    """Un fold con 15 composiciones de test repartidas entre P y U."""
    test = np.r_[np.arange(0, 5), np.arange(30, 40)]
    return Fold(np.setdiff1d(np.arange(60), test), test, ("g",))


BASELINES = {
    "tau": {"kind": "rule", "input": "tau", "rule": {"type": "below", "threshold": 4.18},
            "subset": "tau_defined"},
    "ehull_rule": {"kind": "rule", "input": "ehull", "rule": {"type": "below", "threshold": 0.05},
                   "subset": "post_dft"},
    "ehull_model": {"kind": "model", "estimator": "logistic", "inputs": ["ehull"],
                    "params": {"C": 1.0}, "subset": "post_dft"},
    "sym": {"kind": "model", "estimator": "logistic_onehot",
            "inputs": ["spacegroup_number", "crystal_system"],
            "params": {"C": 1.0, "min_frequency": 2}, "subset": "post_dft"},
    "random": {"kind": "model", "estimator": "random_scores", "inputs": [], "params": {},
               "subset": "all"},
}  # fmt: skip


# ---------------------------------------------------------------------------
# Reglas
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("rule", "expected"),
    [
        ({"type": "below", "threshold": 1.0}, [1, 1, 0, 0]),
        ({"type": "above", "threshold": 1.0}, [0, 1, 1, 1]),
        ({"type": "window", "window": [0.9, 1.6]}, [0, 1, 1, 0]),
    ],
)
def test_rule_scores_match_the_rule(rule, expected):
    x = np.array([0.5, 1.0, 1.5, 2.0])
    scores, thr = rule_scores(x, rule)
    assert (scores >= thr).astype(int).tolist() == expected


def test_window_score_ranks_by_distance_to_center():
    scores, _ = rule_scores(np.array([0.95, 0.80, 1.20]), {"type": "window", "window": [0.9, 1.0]})
    assert scores[0] > scores[1] > scores[2]  # más cerca del centro (0,95) → más alta


def test_invalid_rules_are_rejected():
    with pytest.raises(ValueError):
        rule_scores(np.ones(2), {"type": "window", "window": [1.0, 0.5]})
    with pytest.raises(ValueError):
        rule_scores(np.ones(2), {"type": "otra"})


def test_random_scores_tpr_tracks_ppr():
    # Referencia de azar: con 20 000 casos, la TPR a cualquier umbral ≈ P(ŷ=1).
    s = np.r_[np.ones(10_000, int), np.zeros(10_000, int)]
    p = RandomScores(random_state=1).fit(np.zeros((len(s), 1)), s).predict_proba(s)[:, 1]
    pred = p >= 0.7
    assert abs(pred[s == 1].mean() - pred.mean()) < 0.02


# ---------------------------------------------------------------------------
# Folds y semillas
# ---------------------------------------------------------------------------
def test_frozen_folds_train_is_complement():
    comps = ["A", "B", "C", "D", "E"]
    table = pd.DataFrame({"scheme": "random", "seed": 0, "fold": [0, 0, 1, 1, 1],
                          "composition": ["A", "C", "B", "D", "E"], "group": None})  # fmt: skip
    folds = frozen_folds(table, comps, "random", 0)
    assert [f.test_idx.tolist() for f in folds] == [[0, 2], [1, 3, 4]]
    for f in folds:
        assert not set(f.train_idx) & set(f.test_idx)
        assert len(f.train_idx) + len(f.test_idx) == len(comps)


def test_frozen_folds_errors():
    table = pd.DataFrame({"scheme": "random", "seed": 0, "fold": [0],
                          "composition": ["Z"], "group": None})  # fmt: skip
    with pytest.raises(KeyError):
        frozen_folds(table, ["A"], "random", 0)  # composición fuera del pool
    with pytest.raises(KeyError):
        frozen_folds(table, ["Z"], "cluster", 0)  # sin folds para ese esquema


def test_task_seed_is_deterministic_and_specific():
    a = task_seed(0, "model_init", "loeo_B", 3).generate_state(2).tolist()
    assert a == task_seed(0, "model_init", "loeo_B", 3).generate_state(2).tolist()
    assert a != task_seed(0, "model_init", "loeo_B", 4).generate_state(2).tolist()
    assert a != task_seed(0, "model_init", "family", 3).generate_state(2).tolist()
    assert a != task_seed(1, "model_init", "loeo_B", 3).generate_state(2).tolist()
    assert a != task_seed(0, "fold_assignment", "loeo_B", 3).generate_state(2).tolist()


# ---------------------------------------------------------------------------
# Evaluación de un fold (incluida la fuga)
# ---------------------------------------------------------------------------
def test_evaluate_fold_outputs_only_test_rows_of_each_subset(inputs, fold):
    out = evaluate_fold(inputs, inputs["label"].to_numpy(), fold, 0, "random", 0, BASELINES,
                        INNER)  # fmt: skip
    test_comps = set(inputs.index[fold.test_idx])
    assert set(out["composition"]) <= test_comps
    eligible = set(inputs.index[inputs["post_dft"]])
    for b in ("ehull_rule", "ehull_model", "sym"):
        assert set(out.loc[out["baseline"] == b, "composition"]) == test_comps & eligible
    assert set(out.loc[out["baseline"] == "tau", "composition"]) == test_comps


def test_model_threshold_uses_only_the_training_rows(inputs, fold):
    # El umbral publicado debe coincidir con el recalculado SOLO con el train del fold.
    labels = inputs["label"].to_numpy()
    out = evaluate_fold(inputs, labels, fold, 2, "random", 0, BASELINES, INNER)
    thr_out = out.loc[out["baseline"] == "ehull_model", "threshold"].unique()
    assert len(thr_out) == 1
    from perovskite_pu.utils.seeds import sklearn_random_state

    mask = inputs["post_dft"].to_numpy()
    tr = fold.train_idx[mask[fold.train_idx]]
    est = make_estimator("logistic", ["ehull"], {"C": 1.0},
                         sklearn_random_state(task_seed(0, "model_init", "random", 2)))  # fmt: skip
    X = inputs[["ehull"]].to_numpy()
    inner_rs = sklearn_random_state(task_seed(0, "fold_assignment", "random", 2))
    thr = inner_threshold(est, X[tr], labels[tr], 3, 0.90, inner_rs)
    assert thr_out[0] == pytest.approx(thr)
    # Cambiar el test (etiquetas y entradas) no altera el umbral: no se usa para calibrar.
    tampered = inputs.copy()
    tampered.iloc[fold.test_idx, tampered.columns.get_loc("ehull")] = 99.0
    labels2 = labels.copy()
    labels2[fold.test_idx] = 1 - labels2[fold.test_idx]
    out2 = evaluate_fold(tampered, labels2, fold, 2, "random", 0, BASELINES, INNER)
    assert out2.loc[out2["baseline"] == "ehull_model", "threshold"].iloc[0] == pytest.approx(thr)


def test_evaluate_fold_is_deterministic(inputs, fold):
    labels = inputs["label"].to_numpy()
    a = evaluate_fold(inputs, labels, fold, 1, "cluster", 3, BASELINES, INNER)
    b = evaluate_fold(inputs, labels, fold, 1, "cluster", 3, BASELINES, INNER)
    pd.testing.assert_frame_equal(a, b)


def test_unknown_subset_is_rejected(inputs, fold):
    bad = {"x": {**BASELINES["tau"], "subset": "otro"}}
    with pytest.raises(ValueError):
        evaluate_fold(inputs, inputs["label"].to_numpy(), fold, 0, "random", 0, bad, INNER)


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def test_seed_metrics_and_summary():
    raw = pd.DataFrame({
        "baseline": "b", "scheme": "random", "seed": [0] * 6 + [1] * 6,
        "fold": [0, 0, 0, 1, 1, 1] * 2, "composition": list("ABCDEF") * 2,
        "label": [1, 0, 0, 1, 1, 0] * 2, "score": [0.9, 0.8, 0.1, 0.7, 0.2, 0.3] * 2,
        "threshold": 0.5, "pred": [1, 1, 0, 1, 0, 0] * 2,
    })  # fmt: skip
    m = seed_metrics(raw, [0.34])
    row = m.iloc[0]
    assert row["tpr"] == pytest.approx(2 / 3)  # aciertos 2 de 3 positivos
    assert row["ppr_u"] == pytest.approx(1 / 3)  # B predicha positiva de 3 U
    # P(ŷ=1) fija 0,34 por fold: el mejor de cada fold de 3 (A y D) → 2 de 3 positivos.
    assert row["tpr_at_ppr_0.34"] == pytest.approx(2 / 3)
    assert "roc_auc_lower_bound" in m.columns
    s = summarize(m)
    assert s.loc[0, "tpr_mean"] == pytest.approx(2 / 3) and s.loc[0, "tpr_sd"] == 0.0
    assert s.loc[0, "tpr_ci95_low"] == pytest.approx(2 / 3)
