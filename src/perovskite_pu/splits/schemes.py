"""Generación de folds para cada esquema de partición.

Convenciones de la tabla de entrada (una fila = una composición única):
    composition   fórmula reducida (clave; no puede repetirse)
    A, B          elementos de los sitios A y B
    label         1 = positivo (perovskita en ICSD), 0 = no etiquetado
    year          (opcional) año de la primera entrada en ICSD, para `temporal`

Las composiciones "sintetizadas no perovskita" NO entran aquí: se guardan aparte para
la validación de formabilidad (sección 2.3 de CLAUDE.md).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from perovskite_pu.utils.seeds import sklearn_random_state

# Esquemas soportados, en el orden en que se reportan.
SCHEMES: tuple[str, ...] = (
    "random",
    "group_system",
    "loeo_A",
    "loeo_B",
    "family",
    "cluster",
    "temporal",
)
# Esquemas cuyos folds dependen de la semilla (el resto son deterministas).
# `temporal` es aleatorio solo en la fracción de U que va al test.
RANDOMIZED_SCHEMES: frozenset[str] = frozenset({"random", "group_system", "cluster", "temporal"})


class LeakageError(AssertionError):
    """Se detectó solapamiento de composiciones o de grupos entre train y test."""


@dataclass(frozen=True)
class Fold:
    """Un fold: índices posicionales de train y test, y grupos reservados en test."""

    train_idx: np.ndarray
    test_idx: np.ndarray
    test_groups: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Asignación de grupos
# ---------------------------------------------------------------------------
def assign_b_family(elements: pd.Series, b_families: Mapping[str, Any]) -> pd.Series:
    """Asigna a cada elemento B su familia química según ``configs/data.yaml``.

    Los elementos no listados reciben ``b_families["default_family"]``.
    """
    default = b_families.get("default_family", "main_group")
    lookup = {
        el: fam for fam, members in b_families.items() if fam != "default_family" for el in members
    }
    return elements.map(lambda el: lookup.get(el, default)).astype(str)


def merge_small_groups(
    groups: pd.Series, labels: pd.Series, fallback: pd.Series, min_positives: int
) -> pd.Series:
    """Fusiona los grupos con menos de ``min_positives`` positivos en su grupo de respaldo.

    Ejemplo: en ``loeo_B``, un elemento B con 2 positivos pasa a "family:3d".
    """
    pos_per_group = labels.groupby(groups).sum()
    small = set(pos_per_group[pos_per_group < min_positives].index)
    merged = groups.where(~groups.isin(small), "family:" + fallback.astype(str))
    return merged.astype(str)


def cluster_groups(features: np.ndarray, n_clusters: int, rng: np.random.Generator) -> pd.Series:
    """Grupos por k-means sobre features composicionales (LOCO-CV, Meredig et al., 2018).

    Como en Meredig et al., las features se estandarizan antes de k-means. El clustering es no
    supervisado (no usa etiquetas), por eso se ajusta con todas las composiciones del pool,
    como en el LOCO-CV original.
    """
    seed = sklearn_random_state(np.random.SeedSequence(rng.integers(2**32)))
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    labels = km.fit_predict(StandardScaler().fit_transform(features))
    return pd.Series([f"cluster_{c}" for c in labels])


# ---------------------------------------------------------------------------
# Generadores de folds
# ---------------------------------------------------------------------------
def random_folds(n: int, n_splits: int, rng: np.random.Generator) -> list[Fold]:
    """K-fold aleatorio sobre composiciones (referencia para medir la fuga)."""
    perm = rng.permutation(n)
    folds = []
    for test_idx in np.array_split(perm, n_splits):
        train_idx = np.setdiff1d(perm, test_idx)
        folds.append(Fold(np.sort(train_idx), np.sort(test_idx)))
    return folds


def group_kfold(groups: pd.Series, n_splits: int, rng: np.random.Generator) -> list[Fold]:
    """K-fold por grupos, con los grupos barajados por la semilla."""
    groups = groups.reset_index(drop=True)
    unique = np.array(sorted(groups.unique()))
    if len(unique) < n_splits:
        raise ValueError(f"Hay {len(unique)} grupos, menos que n_splits={n_splits}.")
    folds = []
    for test_groups in np.array_split(rng.permutation(unique), n_splits):
        mask = groups.isin(test_groups).to_numpy()
        folds.append(Fold(np.flatnonzero(~mask), np.flatnonzero(mask), tuple(sorted(test_groups))))
    return folds


def leave_one_group_out(groups: pd.Series) -> list[Fold]:
    """Un fold por grupo (determinista: la semilla no influye en los folds)."""
    groups = groups.reset_index(drop=True)
    folds = []
    for g in sorted(groups.unique()):
        mask = (groups == g).to_numpy()
        folds.append(Fold(np.flatnonzero(~mask), np.flatnonzero(mask), (g,)))
    return folds


def temporal_cutoff(years: pd.Series, target_test_fraction: float) -> int:
    """Año de corte tal que la fracción de positivas con año ≥ corte se acerque al objetivo.

    `years` son los años de primera caracterización de las POSITIVAS (sin NaN). Entre los años
    candidatos se elige el que deja la fracción de test más cercana al objetivo (empate: el
    más reciente).
    """
    y = pd.Series(years).dropna().astype(int)
    if y.empty:
        raise ValueError("No hay años para fijar el corte temporal.")
    candidates = sorted(y.unique())
    return int(min(candidates, key=lambda c: (abs((y >= c).mean() - target_test_fraction), -c)))


def temporal_split(
    years: pd.Series,
    labels: pd.Series,
    cutoff_year: int,
    u_test_fraction: float,
    rng: np.random.Generator,
) -> list[Fold]:
    """Un solo fold: test = positivas con año ≥ corte + una fracción aleatoria de U.

    Las positivas posteriores al corte NO entran al entrenamiento ni siquiera como no
    etiquetadas (regla 2.1.3: nunca la misma composición en train y test). Una fracción
    `u_test_fraction` de U, elegida con `rng`, va al test para poder medir P(ŷ=1|U).
    """
    years = years.reset_index(drop=True)
    labels = labels.reset_index(drop=True).astype(int)
    late_pos = ((labels == 1) & (years >= cutoff_year).fillna(False)).to_numpy(dtype=bool)
    u_idx = np.flatnonzero(labels.to_numpy() == 0)
    n_u = round(u_test_fraction * len(u_idx))
    u_test = rng.choice(u_idx, size=n_u, replace=False) if n_u else np.array([], dtype=int)
    test_mask = late_pos.copy()
    test_mask[u_test] = True
    return [Fold(np.flatnonzero(~test_mask), np.flatnonzero(test_mask), (f">={cutoff_year}",))]


def scheme_groups(
    df: pd.DataFrame,
    scheme: str,
    split_cfg: Mapping[str, Any],
    b_families: Mapping[str, Any],
    rng: np.random.Generator | None = None,
    features: np.ndarray | None = None,
) -> pd.Series | None:
    """Etiqueta de grupo de cada composición según el esquema (``None`` para random/temporal)."""
    df = df.reset_index(drop=True)
    if scheme in ("random", "temporal"):
        return None
    if scheme == "group_system":
        # Sistema químico A–B ordenado alfabéticamente (p. ej. "Sr-Ti").
        return df.apply(lambda r: "-".join(sorted((r["A"], r["B"]))), axis=1).astype(str)
    if scheme in ("loeo_A", "loeo_B"):
        site = scheme[-1]
        groups = df[site].astype(str)
        cfg = split_cfg.get(scheme, {})
        if cfg.get("small_group_policy") == "merge_family":
            fallback = assign_b_family(df["B"], b_families)
            groups = merge_small_groups(
                groups, df["label"], fallback, cfg.get("min_positives_per_element", 5)
            )
        return groups
    if scheme == "family":
        return assign_b_family(df["B"], b_families)
    if scheme == "cluster":
        if features is None or rng is None:
            raise ValueError("El esquema `cluster` necesita `features` y `rng`.")
        return cluster_groups(features, split_cfg["cluster"]["n_clusters"], rng)
    raise ValueError(f"Esquema desconocido: {scheme!r}. Opciones: {SCHEMES}")


def make_folds(
    df: pd.DataFrame,
    scheme: str,
    split_cfg: Mapping[str, Any],
    b_families: Mapping[str, Any],
    rng: np.random.Generator | None = None,
    features: np.ndarray | None = None,
) -> list[Fold]:
    """Genera los folds de un esquema y verifica que no haya fuga antes de devolverlos.

    Args:
        df: tabla de composiciones (ver docstring del módulo).
        scheme: uno de :data:`SCHEMES`.
        split_cfg: sección ``splits`` de ``configs/data.yaml``.
        b_families: sección ``b_families`` de ``configs/data.yaml``.
        rng: flujo ``fold_assignment`` de la semilla (obligatorio en esquemas aleatorios).
        features: matriz de features para el esquema ``cluster``.
    """
    df = df.reset_index(drop=True)
    if df["composition"].duplicated().any():
        dups = df.loc[df["composition"].duplicated(), "composition"].tolist()
        raise LeakageError(f"Composiciones repetidas (deduplicar antes de particionar): {dups[:5]}")
    if scheme in RANDOMIZED_SCHEMES and rng is None:
        raise ValueError(f"El esquema {scheme!r} es aleatorio y necesita `rng`.")

    groups = scheme_groups(df, scheme, split_cfg, b_families, rng=rng, features=features)
    if scheme == "random":
        folds = random_folds(len(df), split_cfg["random"]["n_splits"], rng)
    elif scheme == "group_system":
        folds = group_kfold(groups, split_cfg["group_system"]["n_splits"], rng)
    elif scheme == "temporal":
        tcfg = split_cfg["temporal"]
        if "year" not in df.columns:
            raise ValueError("`temporal` requiere la columna `year` (primera caracterización).")
        frac = tcfg.get("target_test_fraction", 0.15)
        cutoff = tcfg.get("cutoff_year") or temporal_cutoff(df.loc[df["label"] == 1, "year"], frac)
        folds = temporal_split(df["year"], df["label"], int(cutoff), frac, rng)
    else:
        folds = leave_one_group_out(groups)

    assert_no_leakage(folds, df["composition"], groups)
    return folds


# ---------------------------------------------------------------------------
# Verificaciones e informes
# ---------------------------------------------------------------------------
def assert_no_leakage(
    folds: Sequence[Fold], compositions: pd.Series, groups: pd.Series | None = None
) -> None:
    """Falla si algún fold comparte composiciones (o grupos) entre train y test.

    Se ejecuta siempre en :func:`make_folds` y también en ``tests/test_leakage.py``.
    """
    comps = compositions.reset_index(drop=True).to_numpy()
    grp = groups.reset_index(drop=True).to_numpy() if groups is not None else None
    for i, fold in enumerate(folds):
        if np.intersect1d(fold.train_idx, fold.test_idx).size:
            raise LeakageError(f"Fold {i}: índices compartidos entre train y test.")
        shared = set(comps[fold.train_idx]) & set(comps[fold.test_idx])
        if shared:
            raise LeakageError(f"Fold {i}: composiciones en train y test: {sorted(shared)[:5]}")
        if grp is not None:
            shared_g = set(grp[fold.train_idx]) & set(grp[fold.test_idx])
            if shared_g:
                raise LeakageError(f"Fold {i}: grupos en train y test: {sorted(shared_g)[:5]}")


def group_size_report(groups: pd.Series, labels: pd.Series) -> pd.DataFrame:
    """Tamaño de cada grupo y número de positivos (se reporta junto a cada esquema)."""
    frame = pd.DataFrame({"group": groups.to_numpy(), "label": labels.to_numpy()})
    report = frame.groupby("group")["label"].agg(n_compositions="size", n_positives="sum")
    return report.reset_index().sort_values("n_compositions", ascending=False, ignore_index=True)
