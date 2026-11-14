"""Informe diagnóstico de F1.7: particiones sobre el pool PU completo (antes del holdout).

Las particiones DEFINITIVAS se generan sobre `train_pool` después de congelar el holdout
(F1.8), por la regla 2.1.5. Este informe sirve para revisar el diseño antes de congelar.

Escribe en `results/f1_7_particiones/`: summary.json, groups.csv, informe.md.

Uso:
    uv run python scripts/report_f1_7.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from perovskite_pu.evaluation.extrapolation import interpolation_threshold, nearest_train_distances
from perovskite_pu.splits.build import build_pool
from perovskite_pu.splits.schemes import SCHEMES, make_folds, scheme_groups, temporal_cutoff
from perovskite_pu.utils.paths import (
    CONFIGS_DIR,
    INTERIM_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    RESULTS_DIR,
    TRAIN_POOL_FILENAME,
)
from perovskite_pu.utils.run_metadata import git_commit, git_is_dirty
from perovskite_pu.utils.seeds import load_seed_config, rng_streams, sklearn_random_state

OUT = RESULTS_DIR / "f1_7_particiones"


def describe(v: pd.Series) -> str:
    return f"{int(v.min())} / {float(v.median()):g} / {int(v.max())}"


def main() -> None:
    cfg = yaml.safe_load((CONFIGS_DIR / "data.yaml").read_text("utf-8"))
    sc, fam = cfg["splits"], cfg["b_families"]
    seeds = load_seed_config().seeds
    tp_path = PROCESSED_DIR / TRAIN_POOL_FILENAME
    if tp_path.exists():  # particiones definitivas (tras F1.8): solo train_pool
        pool, pool_name = (
            pd.read_parquet(tp_path),
            "train_pool (definitivo, tras congelar el holdout)",
        )
    else:  # diagnóstico previo al holdout
        pool = build_pool(
            pd.read_parquet(INTERIM_DIR / "f1_5_labels.parquet"),
            pd.read_parquet(INTERIM_DIR / "f1_5_entries.parquet"),
            pd.read_parquet(RAW_DIR / "mp_abo3_provenance.parquet"),
            fam,
        )
        pool_name = "pool PU completo (diagnóstico, antes del holdout)"
    X = pd.read_parquet(INTERIM_DIR / "magpie.parquet").loc[pool["composition"]].to_numpy()

    # 1. Grupos por esquema (semilla 0).
    rows, folds0 = [], {}
    for scheme in SCHEMES:
        folds = make_folds(pool, scheme, sc, fam, rng=rng_streams(0)["fold_assignment"], features=X)
        folds0[scheme] = folds
        groups = scheme_groups(
            pool, scheme, sc, fam, rng=rng_streams(0)["fold_assignment"], features=X
        )
        test_n = pd.Series([len(f.test_idx) for f in folds])
        test_p = pd.Series([int(pool["label"].iloc[f.test_idx].sum()) for f in folds])
        g_size = groups.value_counts() if groups is not None else pd.Series(dtype=int)
        rows.append({
            "scheme": scheme, "folds": len(folds), "groups": int(g_size.size) if groups is not None else None,
            "group_size_min_med_max": describe(g_size) if groups is not None else "—",
            "test_size_min_med_max": describe(test_n), "test_pos_min_med_max": describe(test_p),
            "folds_without_positives": int((test_p == 0).sum()),
        })  # fmt: skip
    table = pd.DataFrame(rows)

    # 2. group_system frente a random: en ABO3, cada sistema A–B contiene UNA composición.
    gs = scheme_groups(
        pool, "group_system", sc, fam, rng=rng_streams(0)["fold_assignment"], features=X
    )
    gs_sizes = gs.value_counts().value_counts().sort_index().to_dict()

    # 3. Silueta de k-means (MAGPIE estandarizado), 10 semillas: solo descriptivo.
    Xs = StandardScaler().fit_transform(X)
    sil = {}
    for k in sc["cluster"]["k_scan"]:
        vals = [silhouette_score(Xs, KMeans(n_clusters=k, n_init=10, random_state=sklearn_random_state(
            rng_streams(s)["fold_assignment"].bit_generator.seed_seq)).fit_predict(Xs)) for s in seeds]  # fmt: skip
        sil[k] = [float(np.mean(vals)), float(np.std(vals))]

    # 4. Temporal.
    cutoff = temporal_cutoff(
        pool.loc[pool["label"] == 1, "year"], sc["temporal"]["target_test_fraction"]
    )
    late = int(((pool["label"] == 1) & (pool["year"] >= cutoff)).sum())

    # 5. S1: ¿extrapolan? Distancia al vecino de train más cercano (PCA 95 %, ajustada en train).
    pca = sc["extrapolation"]["pca_variance"]
    d_rand = nearest_train_distances(X, folds0["random"], pca_variance=pca)
    thr = interpolation_threshold(d_rand, sc["extrapolation"]["reference_quantile"])
    s1 = {}
    for scheme in ("loeo_A", "loeo_B", "family", "cluster"):
        groups = scheme_groups(
            pool, scheme, sc, fam, rng=rng_streams(0)["fold_assignment"], features=X
        )
        d = nearest_train_distances(X, folds0[scheme], pca_variance=pca)
        g = pd.DataFrame({"g": groups.to_numpy(), "d": d, "p": pool["label"].to_numpy()})
        per = g.groupby("g").agg(med=("d", "median"), pos=("p", "sum"))
        per["ext"] = per["med"] > thr
        s1[scheme] = {"median_distance": float(np.nanmedian(d)),
                      "groups_extrapolative_frac": float(per["ext"].mean()),
                      "positives_in_extrapolative_frac": float(per.loc[per["ext"], "pos"].sum() / per["pos"].sum())}  # fmt: skip

    s = {
        "code_commit": git_commit(), "code_dirty": git_is_dirty(),
        "pool": {"name": pool_name, "n": len(pool), "positives": int(pool["label"].sum())},
        "schemes": table.to_dict("records"), "group_system_group_size_counts": gs_sizes,
        "silhouette_k_scan": sil, "cluster_k": sc["cluster"]["n_clusters"],
        "temporal": {"cutoff_year": cutoff, "test_positives": late, "test_positive_frac": late / int(pool["label"].sum())},
        "s1_reference": {"random_median_distance": float(np.nanmedian(d_rand)), "q95_threshold": float(thr)},
        "s1": s1,
    }  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(s, indent=2, ensure_ascii=False) + "\n", "utf-8")
    table.to_csv(OUT / "groups.csv", index=False, lineterminator="\n")

    s1_rows = "\n".join(
        f"| `{k}` | {v['median_distance']:.2f} | {v['groups_extrapolative_frac']:.0%} | {v['positives_in_extrapolative_frac']:.0%} |"
        for k, v in s1.items()
    )
    sil_rows = " · ".join(f"k={k}: {m:.3f}±{sd:.3f}" for k, (m, sd) in sil.items())
    md = f"""# Informe F1.7: particiones

> Generado por `scripts/report_f1_7.py` (commit `{s["code_commit"][:7]}`{", árbol sucio" if s["code_dirty"] else ""}).
> No editar a mano. **Pool: {pool_name}: {len(pool)} composiciones, {s["pool"]["positives"]} P.**
> Las particiones definitivas usan solo `train_pool` (regla 2.1.5); están en
> `data/splits/folds.parquet`, con su hash en `HASHES.json`.

## 1. Implementación

Implementación propia en `src/perovskite_pu/splits/` y **no MatFold** (motivo en la bitácora):
MatFold no tiene particiones por sitio A/B (`loeo_A`/`loeo_B`), por familia del elemento B, ni
LOCO por k-means o temporal, y exige una estructura por entrada, mientras que nuestras
etiquetas son por composición. Lo que MatFold sí ofrece (`random`, `chemsys`) coincide con
nuestro `random` y `group_system`.

## 2. Esquemas (semilla 0)

{table.rename(columns={"scheme": "esquema", "folds": "folds", "groups": "grupos", "group_size_min_med_max": "tamaño de grupo (mín/mediana/máx)", "test_size_min_med_max": "tamaño de test (mín/mediana/máx)", "test_pos_min_med_max": "P en test (mín/mediana/máx)", "folds_without_positives": "folds sin P"}).to_markdown(index=False)}

- **`loeo_B`:** los elementos B con < {sc["loeo_B"]["min_positives_per_element"]} positivas se fusionan por familia.
- **`loeo_A`:** se reportan aparte (`report_separately`); hay folds sin positivas, que solo
  aportan P(ŷ=1|U). Por eso la TPR se micro-agrega.

## 3. `group_system` frente a `random`

Tamaños de grupo de `group_system` (tamaño: n.º de grupos): {gs_sizes}. **En ABO₃, cada
sistema químico A–B contiene exactamente una composición**, así que `group_system` es,
por construcción, idéntico a `random`. Se mantiene solo como diagnóstico, como prevé la
regla 2.1.4.

## 4. `cluster`: silueta de k-means sobre MAGPIE estandarizado (10 semillas)

{sil_rows}

La silueta es plana: no hay un k natural. Se usa **k = {s["cluster_k"]}** (decisión de la autora).

## 5. `temporal`

Corte: **{cutoff}** (fracción objetivo {sc["temporal"]["target_test_fraction"]:.0%}). Test: {late} positivas ({late / s["pool"]["positives"]:.1%}) caracterizadas
desde {cutoff}, más la misma fracción de U, elegida con la semilla. Limitación: la positiva más
reciente es de 2013, porque la copia de ICSD que usa MP no cubre años posteriores.

## 6. S1: ¿extrapolan los esquemas? (preregistro §10b)

Referencia `random`: mediana de la distancia al vecino de train más cercano
{s["s1_reference"]["random_median_distance"]:.2f}; umbral (q95) {thr:.2f}. Espacio: MAGPIE estandarizado + PCA al 95 %,
ajustados en train.

| Esquema | Distancia mediana | Grupos extrapolativos | Positivas en grupos extrapolativos |
|---|---|---|---|
{s1_rows}

Interpretación preregistrada: si < 50 % de las positivas de un esquema están en grupos
extrapolativos, sus resultados de H1 se describen como de "extrapolación débil".
"""
    (OUT / "informe.md").write_text(md, "utf-8")
    print(
        json.dumps(
            {
                k: s[k]
                for k in (
                    "schemes",
                    "group_system_group_size_counts",
                    "temporal",
                    "s1_reference",
                    "s1",
                )
            },
            indent=1,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":  # requisito en Windows (regla 2.2)
    main()
