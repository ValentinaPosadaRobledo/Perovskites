"""Tests de utilidades: semillas, hashing canónico, metadatos de corrida y configuraciones."""

from __future__ import annotations

import subprocess

import numpy as np
import pandas as pd
import pytest
import yaml

from perovskite_pu.utils.hashing import (
    canonical_csv_bytes,
    config_hash,
    sha256_file,
    write_canonical_csv,
)
from perovskite_pu.utils.paths import CONFIGS_DIR
from perovskite_pu.utils.run_metadata import (
    DirtyTreeError,
    build_run_metadata,
    ensure_clean_tree,
    write_run_metadata,
)
from perovskite_pu.utils.seeds import (
    STREAM_NAMES,
    load_seed_config,
    rng_streams,
    sklearn_random_state,
    spawn_streams,
)


# ---------------------------------------------------------------------------
# Semillas
# ---------------------------------------------------------------------------
def test_project_seed_config():
    """configs/seeds.yaml: 10 semillas 0–9 y holdout_seed 2026 independiente."""
    cfg = load_seed_config()
    assert cfg.seeds == tuple(range(10))
    assert cfg.holdout_seed == 2026
    assert cfg.holdout_seed not in cfg.seeds
    assert cfg.bootstrap_seed == 20261113  # fijada en el preregistro (sección 9)
    assert cfg.bootstrap_seed not in (*cfg.seeds, cfg.holdout_seed)
    assert cfg.active_seeds(dev_mode=True) == (0, 1, 2)


@pytest.mark.parametrize(
    "content",
    [
        "seeds: [0, 1, 1]\nholdout_seed: 2026\nbootstrap_seed: 7\n",  # repetidas
        "seeds: [0, 1, 2]\nholdout_seed: 1\nbootstrap_seed: 7\n",  # holdout_seed coincide
        "seeds: [0, 1, 2]\nholdout_seed: 2026\nbootstrap_seed: 2\n",  # bootstrap coincide
        "seeds: [0, 1, 2]\nholdout_seed: 2026\nbootstrap_seed: 2026\n",  # iguales entre sí
        "seeds: [0, 1, 2]\nholdout_seed: 2026\n",  # falta bootstrap_seed
        "seeds: []\nholdout_seed: 2026\nbootstrap_seed: 7\n",  # vacía
        "seeds: [0, -1]\nholdout_seed: 2026\nbootstrap_seed: 7\n",  # negativa
        "seeds: [0, 1]\nholdout_seed: 2026\nbootstrap_seed: 7\ndev_n_seeds: 5\n",  # > len
    ],
)
def test_seed_config_validation(tmp_path, content):
    path = tmp_path / "seeds.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_seed_config(path)


def test_streams_are_reproducible_and_independent():
    """Misma semilla -> mismos flujos; los tres flujos de una semilla son distintos entre sí."""
    a, b = rng_streams(3), rng_streams(3)
    assert tuple(a) == STREAM_NAMES
    draws_a = {k: g.random(5) for k, g in a.items()}
    draws_b = {k: g.random(5) for k, g in b.items()}
    for k in STREAM_NAMES:
        np.testing.assert_array_equal(draws_a[k], draws_b[k])
    assert not np.allclose(draws_a["fold_assignment"], draws_a["pu_sampling"])
    assert not np.allclose(draws_a["pu_sampling"], draws_a["model_init"])
    # Semillas distintas -> flujos distintos
    assert not np.allclose(rng_streams(4)["fold_assignment"].random(5), draws_a["fold_assignment"])


def test_sklearn_random_state_is_valid_int():
    rs = sklearn_random_state(spawn_streams(0)["model_init"])
    assert isinstance(rs, int) and 0 <= rs < 2**32
    assert rs == sklearn_random_state(spawn_streams(0)["model_init"])


# ---------------------------------------------------------------------------
# Hashing canónico
# ---------------------------------------------------------------------------
def test_canonical_csv_is_order_independent_and_lf():
    df = pd.DataFrame({"composition": ["SrTiO3", "BaTiO3", "CaTiO3"], "label": [1, 1, 0]})
    shuffled = df.sample(frac=1, random_state=1)[["label", "composition"]]
    a = canonical_csv_bytes(df, ["composition"])
    b = canonical_csv_bytes(shuffled, ["composition"])
    assert a == b
    assert b"\r" not in a
    assert a.splitlines()[1].startswith(b"BaTiO3")


def test_write_canonical_csv_hash_matches_file(tmp_path):
    df = pd.DataFrame({"composition": ["SrTiO3", "BaTiO3"]})
    path = tmp_path / "x.csv"
    assert write_canonical_csv(df, path, ["composition"]) == sha256_file(path)


def test_config_hash_ignores_key_order():
    assert config_hash({"a": 1, "b": [1, 2]}) == config_hash({"b": [1, 2], "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


# ---------------------------------------------------------------------------
# Metadatos de corrida y árbol de git
# ---------------------------------------------------------------------------
@pytest.fixture
def tmp_repo(tmp_path):
    """Repositorio git temporal con un commit inicial."""

    def git(*args):
        subprocess.run(
            ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

    git("init", "-q")
    (tmp_path / "a.txt").write_text("hola\n", encoding="utf-8")
    git("add", "a.txt")
    git("commit", "-q", "-m", "inicial")
    return tmp_path


def test_clean_tree_passes(tmp_repo):
    assert ensure_clean_tree(tmp_repo) is None


def test_dirty_tree_is_refused_unless_allowed(tmp_repo):
    (tmp_repo / "a.txt").write_text("cambio\n", encoding="utf-8")
    with pytest.raises(DirtyTreeError):
        ensure_clean_tree(tmp_repo)
    diff = ensure_clean_tree(tmp_repo, allow_dirty=True)
    assert "cambio" in diff


def test_run_metadata_records_required_fields(tmp_repo):
    (tmp_repo / "nuevo.txt").write_text("x\n", encoding="utf-8")  # árbol sucio
    meta = build_run_metadata(
        {"name": "demo"},
        [0, 1],
        n_jobs=2,
        mp_database_version="2026.x",
        allow_dirty=True,
        repo=tmp_repo,
    )
    for key in (
        "git_commit",
        "config_hash",
        "seeds",
        "package_versions",
        "mp_database_version",
        "cpu_count",
        "n_jobs",
        "timestamp_utc",
        "uv_lock_sha256",
    ):
        assert key in meta, key
    assert meta["git_dirty"] is True
    out = write_run_metadata(meta, tmp_repo)
    assert out.exists()
    assert "nuevo.txt" in (tmp_repo / "git_diff.patch").read_text("utf-8")


# ---------------------------------------------------------------------------
# Configuraciones
# ---------------------------------------------------------------------------
def test_b_families_are_disjoint(data_cfg):
    fams = {k: v for k, v in data_cfg["b_families"].items() if k != "default_family"}
    seen: dict[str, str] = {}
    for fam, elements in fams.items():
        for el in elements:
            assert el not in seen, f"{el} está en {seen.get(el)} y en {fam}"
            seen[el] = fam


def test_experiment_template_has_no_seeds():
    """Las semillas solo viven en configs/seeds.yaml, nunca en un YAML de experimento."""
    for path in (CONFIGS_DIR / "experiments").glob("*.yaml"):
        cfg = yaml.safe_load(path.read_text("utf-8"))
        text = yaml.safe_dump(cfg)
        assert "seeds:" not in text and "random_state" not in text, path.name
