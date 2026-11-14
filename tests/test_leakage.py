"""Tests de fuga de información (regla 2.1.9 de CLAUDE.md).

Capas:
1. SIEMPRE ACTIVOS (datos sintéticos de tests/fixtures/):
   - sin solapamiento de composiciones entre train y test en ningún fold de ningún esquema;
   - sin solapamiento de grupos en los esquemas por grupos;
   - el preprocesamiento se ajusta solo con train;
   - análisis estático (ast): training/ no referencia el holdout;
   - audit hook: una corrida sintética nunca abre holdout_frozen.csv;
   - load_holdout() exige HOLDOUT_UNLOCK=1 y verifica el hash;
   - auditoría de features (F2.3): la configuración de familias cumple
     `configs/feature_audit.yaml` (lista blanca, patrones prohibidos, pre_dft limpia).
2. DEPENDIENTES DE DATOS (se saltan sin datos; fallan con REQUIRE_DATA=1 o si
   data/processed/ tiene datos): holdout ∩ train_pool = ∅, hash canónico correcto y familias
   de features calculadas sin columnas prohibidas ni sin auditar.

Ejecutar siempre con ``pytest -rs`` (ya está en pyproject.toml).
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from perovskite_pu.evaluation.holdout import (
    HoldoutIntegrityError,
    HoldoutLockedError,
    load_holdout,
)
from perovskite_pu.splits import SCHEMES, Fold, LeakageError, assert_no_leakage, make_folds
from perovskite_pu.splits.schemes import RANDOMIZED_SCHEMES, scheme_groups
from perovskite_pu.training.cv import fit_predict_folds
from perovskite_pu.utils.hashing import canonical_csv_bytes, read_hashes, sha256_file
from perovskite_pu.utils.paths import (
    HASHES_FILENAME,
    HOLDOUT_FILENAME,
    PROCESSED_DIR,
    ROOT,
    SPLITS_DIR,
    TRAIN_POOL_FILENAME,
)
from perovskite_pu.utils.seeds import rng_streams

TRAINING_DIR = ROOT / "src" / "perovskite_pu" / "training"
TEST_SEEDS = (0, 1, 2)  # 3 semillas bastan para los tests de mecánica
FEATURES = ["f0", "f1", "f2", "f3"]
# Esquemas que reparten TODAS las composiciones en test exactamente una vez.
PARTITION_SCHEMES = tuple(s for s in SCHEMES if s != "temporal")


def _folds(toy_df, scheme, split_cfg, b_families, toy_features, seed=0) -> list[Fold]:
    """Atajo: genera los folds de un esquema con el flujo `fold_assignment` de la semilla."""
    rng = rng_streams(seed)["fold_assignment"]
    return make_folds(toy_df, scheme, split_cfg, b_families, rng=rng, features=toy_features)


# ===========================================================================
# 1a. Particiones: composiciones y grupos
# ===========================================================================
def test_toy_compositions_are_unique(toy_df):
    """Precondición: la etiqueta es a nivel de composición, sin duplicados."""
    assert not toy_df["composition"].duplicated().any()


@pytest.mark.parametrize("seed", TEST_SEEDS)
@pytest.mark.parametrize("scheme", SCHEMES)
def test_no_composition_overlap(scheme, seed, toy_df, split_cfg, b_families, toy_features):
    """Ninguna composición aparece en train y test del mismo fold (verificación independiente)."""
    folds = _folds(toy_df, scheme, split_cfg, b_families, toy_features, seed)
    assert folds, "El esquema no generó folds"
    comps = toy_df["composition"].to_numpy()
    for fold in folds:
        assert fold.test_idx.size > 0
        assert not set(comps[fold.train_idx]) & set(comps[fold.test_idx])


@pytest.mark.parametrize("seed", TEST_SEEDS)
@pytest.mark.parametrize("scheme", [s for s in SCHEMES if s not in ("random", "temporal")])
def test_no_group_overlap(scheme, seed, toy_df, split_cfg, b_families, toy_features):
    """En los esquemas por grupos, ningún grupo aparece a la vez en train y test."""
    rng = rng_streams(seed)["fold_assignment"]
    folds = make_folds(toy_df, scheme, split_cfg, b_families, rng=rng, features=toy_features)
    # Se recalculan los grupos con un generador idéntico para no depender de make_folds.
    rng2 = rng_streams(seed)["fold_assignment"]
    groups = scheme_groups(toy_df, scheme, split_cfg, b_families, rng=rng2, features=toy_features)
    g = groups.to_numpy()
    for fold in folds:
        assert not set(g[fold.train_idx]) & set(g[fold.test_idx])


@pytest.mark.parametrize("scheme", PARTITION_SCHEMES)
def test_each_composition_tested_exactly_once(scheme, toy_df, split_cfg, b_families, toy_features):
    """En CV, cada composición está en test exactamente una vez y train ∪ test = todo."""
    folds = _folds(toy_df, scheme, split_cfg, b_families, toy_features)
    counts = np.zeros(len(toy_df), dtype=int)
    for fold in folds:
        counts[fold.test_idx] += 1
        assert len(fold.train_idx) + len(fold.test_idx) == len(toy_df)
    assert (counts == 1).all()


def test_loeo_b_merges_small_elements_into_family(toy_df, split_cfg, b_families):
    """Los elementos B con < min_positives positivos se fusionan por familia."""
    groups = scheme_groups(toy_df, "loeo_B", split_cfg, b_families)
    min_pos = split_cfg["loeo_B"]["min_positives_per_element"]
    pos = toy_df.groupby("B")["label"].sum()
    for element, n_pos in pos.items():
        g = set(groups[toy_df["B"] == element])
        if n_pos < min_pos:
            assert all(x.startswith("family:") for x in g), element
        else:
            assert g == {element}


@pytest.mark.parametrize("scheme", ["loeo_A", "loeo_B", "family"])
def test_loeo_folds_are_deterministic(scheme, toy_df, split_cfg, b_families, toy_features):
    """En leave-one-group-out la semilla NO cambia los folds (solo afecta bagging y modelo)."""
    f0 = _folds(toy_df, scheme, split_cfg, b_families, toy_features, seed=0)
    f1 = _folds(toy_df, scheme, split_cfg, b_families, toy_features, seed=1)
    assert [tuple(f.test_idx) for f in f0] == [tuple(f.test_idx) for f in f1]


@pytest.mark.parametrize("scheme", sorted(RANDOMIZED_SCHEMES))
def test_randomized_folds_depend_on_seed(scheme, toy_df, split_cfg, b_families, toy_features):
    """En los esquemas aleatorios, semillas distintas dan folds distintos y la misma, iguales."""
    a = _folds(toy_df, scheme, split_cfg, b_families, toy_features, seed=0)
    b = _folds(toy_df, scheme, split_cfg, b_families, toy_features, seed=0)
    c = _folds(toy_df, scheme, split_cfg, b_families, toy_features, seed=1)
    key = lambda folds: [tuple(f.test_idx) for f in folds]  # noqa: E731
    assert key(a) == key(b)
    assert key(a) != key(c)


def test_make_folds_rejects_duplicate_compositions(toy_df, split_cfg, b_families):
    """Si hay composiciones repetidas (no se deduplicó), make_folds se niega a particionar."""
    dup = pd.concat([toy_df, toy_df.iloc[[0]]], ignore_index=True)
    with pytest.raises(LeakageError, match="repetidas"):
        make_folds(dup, "loeo_B", split_cfg, b_families)


def test_assert_no_leakage_detects_overlap(toy_df):
    """Control positivo: el verificador SÍ detecta un fold con fuga."""
    bad = Fold(train_idx=np.array([0, 1, 2]), test_idx=np.array([2, 3]))
    with pytest.raises(LeakageError):
        assert_no_leakage([bad], toy_df["composition"])
    # Fuga por grupo sin repetir composiciones: BaAlO3 (train) y BaCoO3 (test) comparten A=Ba.
    fold = Fold(train_idx=np.array([0]), test_idx=np.array([1]))
    with pytest.raises(LeakageError, match="grupos"):
        assert_no_leakage([fold], toy_df["composition"], toy_df["A"])


# ===========================================================================
# 1b. Preprocesamiento ajustado solo con train
# ===========================================================================
def _toy_pipeline() -> Pipeline:
    """Pipeline mínimo con imputación + escalado + clasificador (como exige la regla 2.1.6)."""
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="mean")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression()),
        ]
    )


@pytest.mark.parametrize("scheme", ["random", "loeo_B"])
def test_preprocessing_fitted_on_train_only(scheme, toy_df, split_cfg, b_families, toy_features):
    """Las estadísticas del imputador y del escalador salen SOLO de las filas de train."""
    folds = _folds(toy_df, scheme, split_cfg, b_families, toy_features)
    X = toy_df[FEATURES]
    results = fit_predict_folds(
        _toy_pipeline(), X, toy_df["label"].to_numpy(), folds, return_estimators=True
    )
    global_mean = X.mean().to_numpy()
    differs_from_global = False
    for fold, res in zip(folds, results, strict=True):
        X_train = X.iloc[fold.train_idx]
        imputer = res.estimator.named_steps["impute"]
        scaler = res.estimator.named_steps["scale"]
        np.testing.assert_allclose(imputer.statistics_, X_train.mean().to_numpy())
        X_train_imputed = X_train.fillna(X_train.mean())
        np.testing.assert_allclose(scaler.mean_, X_train_imputed.mean().to_numpy())
        differs_from_global |= not np.allclose(scaler.mean_, global_mean)
        assert len(res.scores) == len(fold.test_idx)
    # Control: si las medias coincidieran siempre con las globales, el test no probaría nada.
    assert differs_from_global


# ===========================================================================
# 1c. Análisis estático: training/ no puede referenciar el holdout
# ===========================================================================
FORBIDDEN_TOKEN = "holdout"


def find_holdout_references(source: str, filename: str = "<src>") -> list[str]:
    """Busca referencias al holdout en el AST de un archivo.

    Detecta: imports de módulos o nombres que contienen "holdout", nombres de variables,
    atributos, argumentos y definiciones con "holdout", y cadenas de texto (incluidos
    docstrings) que lo mencionen, lo que cubre ``importlib.import_module("...holdout")``
    y rutas como ``"holdout_frozen.csv"``. No mira comentarios (no están en el AST).
    """
    hits: list[str] = []
    tree = ast.parse(source, filename=filename)
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names] + [a.asname or "" for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""] + [a.name for a in node.names]
        elif isinstance(node, ast.Name):
            names = [node.id]
        elif isinstance(node, ast.Attribute):
            names = [node.attr]
        elif isinstance(node, ast.arg):
            names = [node.arg]
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names = [node.name]
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            names = [node.value]
        for name in names:
            if FORBIDDEN_TOKEN in name.lower():
                hits.append(f"{filename}:{getattr(node, 'lineno', '?')}: {name[:60]!r}")
    return hits


def test_static_checker_detects_violations():
    """Control positivo: el analizador estático sí detecta cada forma de referencia."""
    snippets = [
        "from perovskite_pu.evaluation.holdout import load_holdout",
        "import perovskite_pu.evaluation.holdout as h",
        "from perovskite_pu.evaluation import holdout",
        "x = load_holdout()",
        "import importlib; importlib.import_module('perovskite_pu.evaluation.holdout')",
        "path = 'data/splits/holdout_frozen.csv'",
        "def f(holdout_df): pass",
        "from perovskite_pu.utils.paths import HOLDOUT_FILENAME",
    ]
    for snippet in snippets:
        assert find_holdout_references(snippet), snippet
    assert not find_holdout_references("from perovskite_pu.splits import make_folds")


def test_training_package_does_not_reference_holdout():
    """Ningún archivo de src/perovskite_pu/training/ referencia el holdout ni load_holdout."""
    files = sorted(TRAINING_DIR.rglob("*.py"))
    assert files, f"No se encontraron archivos en {TRAINING_DIR}"
    hits = [h for f in files for h in find_holdout_references(f.read_text("utf-8"), str(f))]
    assert not hits, "Referencias prohibidas al holdout en training/:\n" + "\n".join(hits)


# ===========================================================================
# 1d. Audit hook: una corrida sintética completa nunca abre el holdout
# ===========================================================================
def test_audit_hook_blocks_holdout_open(tmp_path, guard_holdout):
    """Control positivo: con el guardián armado, abrir holdout_frozen.csv falla."""
    fake = tmp_path / HOLDOUT_FILENAME
    fake.write_text("composition\nSrTiO3\n", encoding="utf-8")
    with guard_holdout(), pytest.raises(RuntimeError, match="holdout"):
        pd.read_csv(fake)
    pd.read_csv(fake)  # fuera del bloque, la lectura vuelve a estar permitida


def test_synthetic_training_run_never_opens_holdout(
    toy_df, split_cfg, b_families, toy_features, guard_holdout
):
    """Corrida de entrenamiento simulada (esquemas × 3 semillas) con el guardián armado."""
    X, s = toy_df[FEATURES], toy_df["label"].to_numpy()
    with guard_holdout():
        for seed in TEST_SEEDS:
            for scheme in SCHEMES:
                folds = _folds(toy_df, scheme, split_cfg, b_families, toy_features, seed)
                results = fit_predict_folds(_toy_pipeline(), X, s, folds)
                assert len(results) == len(folds)


# ===========================================================================
# 1e. load_holdout(): bloqueo y verificación de integridad (archivos sintéticos)
# ===========================================================================
@pytest.fixture
def fake_holdout(tmp_path):
    """Holdout sintético en forma canónica y su HASHES.json, en un directorio temporal."""
    df = pd.DataFrame({"composition": ["SrTiO3", "BaZrO3"], "label": [1, 0]})
    path = tmp_path / HOLDOUT_FILENAME
    path.write_bytes(canonical_csv_bytes(df, ["composition"]))
    hashes = tmp_path / HASHES_FILENAME
    hashes.write_text(f'{{"files": {{"{HOLDOUT_FILENAME}": "{sha256_file(path)}"}}}}', "utf-8")
    return path, hashes


@pytest.mark.parametrize("value", [None, "0", "true", "yes"])
def test_load_holdout_locked_without_unlock(value, fake_holdout, monkeypatch):
    """Sin HOLDOUT_UNLOCK=1 exacto, load_holdout() se niega a leer."""
    if value is None:
        monkeypatch.delenv("HOLDOUT_UNLOCK", raising=False)
    else:
        monkeypatch.setenv("HOLDOUT_UNLOCK", value)
    with pytest.raises(HoldoutLockedError):
        load_holdout(*fake_holdout)


def test_load_holdout_reads_when_unlocked_and_hash_matches(fake_holdout, monkeypatch):
    monkeypatch.setenv("HOLDOUT_UNLOCK", "1")
    df = load_holdout(*fake_holdout)
    assert sorted(df["composition"]) == ["BaZrO3", "SrTiO3"]


def test_load_holdout_rejects_tampered_file(fake_holdout, monkeypatch):
    """Si el archivo cambia después de registrar su hash, la lectura falla."""
    path, hashes = fake_holdout
    path.write_bytes(path.read_bytes() + b"CaTiO3,1\n")
    monkeypatch.setenv("HOLDOUT_UNLOCK", "1")
    with pytest.raises(HoldoutIntegrityError):
        load_holdout(path, hashes)


def test_load_holdout_requires_registered_hash(fake_holdout, monkeypatch, tmp_path):
    path, _ = fake_holdout
    empty = tmp_path / "empty.json"
    empty.write_text('{"files": {}}', "utf-8")
    monkeypatch.setenv("HOLDOUT_UNLOCK", "1")
    with pytest.raises(HoldoutIntegrityError, match="No hay hash"):
        load_holdout(path, empty)


# ===========================================================================
# 1e. Auditoría de features (F2.3, regla 2.1.8)
# ===========================================================================
def _config_families(data_cfg: dict) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Familias a partir SOLO de la configuración, con una columna ficticia por grupo."""
    from perovskite_pu.features.post_dft import family_columns_post
    from perovskite_pu.features.pre_dft import GEOMETRIC_COLUMNS, family_columns

    fam_cfg = data_cfg["features"]["families"]
    groups = {g for spec in fam_cfg.values() for g in spec["composition"]}
    by_group = {g: [f"MagpieData mean {g}"] for g in groups}  # nombres como los de matminer
    by_group["geometric"] = list(GEOMETRIC_COLUMNS.values())
    pre = family_columns(fam_cfg, by_group)
    post = family_columns_post(data_cfg["features"]["families_post_dft"], pre)
    return {**pre, **post}, by_group


def test_feature_families_config_passes_audit(data_cfg):
    """Las familias de configs/data.yaml cumplen feature_audit.yaml (sin datos)."""
    from perovskite_pu.features.audit import check_family_columns, load_audit_config

    families, by_group = _config_families(data_cfg)
    assert check_family_columns(families, load_audit_config(), by_group) == []


def test_pre_dft_has_no_tau_and_no_thermo(data_cfg):
    families, _ = _config_families(data_cfg)
    assert not any(c == "geom_tau" or c.startswith("thermo_") for c in families["pre_dft"])


def test_audit_detects_forbidden_unaudited_and_impure_pre_dft(data_cfg):
    from perovskite_pu.features.audit import check_family_columns, load_audit_config

    _, by_group = _config_families(data_cfg)
    bad = {
        "pre_dft": ["geom_t", "thermo_ehull"],          # termo en la familia principal
        "post_dft": ["experimental_gga", "columna_x"],  # metadato y columna sin auditar
    }  # fmt: skip
    problems = check_family_columns(bad, load_audit_config(), by_group)
    assert any("prohibida 'experimental_gga'" in p for p in problems)
    assert any("sin grupo auditado 'columna_x'" in p for p in problems)
    assert any("pre_dft: 'thermo_ehull'" in p for p in problems)


@pytest.mark.parametrize(
    "name",
    ["MagpieData mean SpaceGroupNumber", "compound possible", "avg s valence electrons",
     "0-norm", "geom_t", "thermo_ehull"],
)  # fmt: skip
def test_forbidden_patterns_spare_legitimate_features(name):
    import re

    from perovskite_pu.features.audit import load_audit_config

    patterns = [re.compile(p) for p in load_audit_config()["forbidden_patterns"]]
    assert not any(p.search(name) for p in patterns)


# ===========================================================================
# 2. Dependientes de datos (se activan tras F1.8)
# ===========================================================================
HOLDOUT_PATH: Path = SPLITS_DIR / HOLDOUT_FILENAME
HASHES_PATH: Path = SPLITS_DIR / HASHES_FILENAME
TRAIN_POOL_PATH: Path = PROCESSED_DIR / TRAIN_POOL_FILENAME


@pytest.mark.requires_data
def test_holdout_disjoint_from_train_pool(require_files):
    """Ninguna composición del holdout está en train_pool."""
    require_files(HOLDOUT_PATH, TRAIN_POOL_PATH)
    # Lectura directa (sin load_holdout): solo se comparan composiciones, sin evaluar nada.
    holdout = pd.read_csv(HOLDOUT_PATH, usecols=["composition"])
    pool = pd.read_parquet(TRAIN_POOL_PATH, columns=["composition"])
    shared = set(holdout["composition"]) & set(pool["composition"])
    assert not shared, f"Composiciones del holdout en train_pool: {sorted(shared)[:10]}"


@pytest.mark.requires_data
def test_holdout_hash_matches_registry(require_files):
    """El holdout está en forma canónica y su SHA-256 coincide con HASHES.json."""
    require_files(HOLDOUT_PATH, HASHES_PATH)
    expected = read_hashes(HASHES_PATH).get(HOLDOUT_FILENAME)
    assert expected, f"{HOLDOUT_FILENAME} no está registrado en {HASHES_PATH}"
    assert sha256_file(HOLDOUT_PATH) == expected
    # Forma canónica: UTF-8 sin BOM, solo LF y filas ordenadas por composición.
    raw = HOLDOUT_PATH.read_bytes()
    assert b"\r" not in raw, "El holdout contiene CRLF"
    assert not raw.startswith(b"\xef\xbb\xbf"), "El holdout tiene BOM"
    comps = pd.read_csv(HOLDOUT_PATH, usecols=["composition"])["composition"].tolist()
    assert comps == sorted(comps), "El holdout no está ordenado por composición"


@pytest.mark.requires_data
def test_computed_feature_families_pass_audit(require_files, data_cfg):
    """Las familias calculadas (F2.1–F2.2) no tienen columnas prohibidas ni sin auditar."""
    import json

    from perovskite_pu.features.audit import check_family_columns, load_audit_config

    out = data_cfg["features"]["output"]
    pre_path, post_path = ROOT / out["columns"], ROOT / out["post_columns"]
    require_files(pre_path, post_path)
    pre = json.loads(pre_path.read_text("utf-8"))
    post = json.loads(post_path.read_text("utf-8"))
    families = {**pre["families"], **post["families"]}
    assert check_family_columns(families, load_audit_config(), pre["groups"]) == []
