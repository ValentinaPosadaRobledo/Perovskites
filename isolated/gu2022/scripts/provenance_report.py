"""F3.5 (exploratorio): ¿cuánto del rendimiento de Gu et al. explica la procedencia?

Las positivas de Gu son estructuras experimentales (ICSD) y sus no etiquetadas, sobre todo
prototipos hipotéticos. Si un modelo que solo ve la simetría y la distorsión (grupo espacial,
Pm-3m, ángulo B–X–B mínimo; ver `provenance_features.py`) se acerca a su TPR al mismo
P(ŷ=1|U), parte de su rendimiento se explicaría por la procedencia de las estructuras.

Protocolo, el mismo de su ensemble: un modelo por cada una de sus 100 particiones, entrenado
con PTrain+PVal frente a NTrain+NVal; la puntuación de cada cristal es la media de los 100.
Se evalúa en su PTest (94 P, el mismo en todas las particiones) frente a todas las U, igual
que su ensemble (sus U de entrenamiento también entran en P(ŷ=1|U), así que ambos lados están
en las mismas condiciones). Modelos:

- `rule_not_pm3m`: regla fija, positivo si el grupo espacial no es Pm-3m.
- `lr_spacegroup`: regresión logística sobre el grupo espacial (one-hot).
- `rf_symmetry_angle`: Random Forest sobre grupo espacial, Pm-3m y ángulo B–X–B mínimo.
- `lr_x_element`: regresión logística solo sobre el elemento del sitio X (one-hot); mide la
  separación que da la química del anión, sin estructura (en Gu, X no siempre es O o F).
- `lr_abx_elements`: regresión logística sobre los elementos A, B y X (one-hot por sitio),
  sin ninguna información estructural.

Con empates (regla y simetría), el TPR a un P(ŷ=1|U) dado se interpola en la curva ROC:
es el valor esperado al desempatar al azar.

    .venv/bin/python -I scripts/provenance_report.py --repo <repo> --features <csv> --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

FIXED_PPR_U = (0.05, 0.10, 0.20)


def tpr_at_ppr_u(scores_p: np.ndarray, scores_u: np.ndarray, ppr_u: float) -> float:
    """TPR esperado (desempate al azar) en el punto de la ROC con P(ŷ=1|U) = `ppr_u`."""
    y = np.r_[np.ones(len(scores_p)), np.zeros(len(scores_u))]
    fpr, tpr, _ = roc_curve(y, np.r_[scores_p, scores_u], drop_intermediate=False)
    return float(np.interp(ppr_u, fpr, tpr))


def onehot_column(values: pd.Series) -> np.ndarray:
    """Codificación one-hot de una columna categórica (niveles en orden alfabético)."""
    levels = sorted(values.unique())
    return (values.to_numpy()[:, None] == np.array(levels)[None, :]).astype(float)


def design(feat: pd.DataFrame, kind: str, sg_levels: list[int]) -> np.ndarray:
    """Matriz de features según `kind`: simetría, simetría + ángulo, elemento X o A, B y X."""
    if kind == "x_element":
        return onehot_column(feat["X"])
    if kind == "abx_elements":
        return np.c_[onehot_column(feat["A"]), onehot_column(feat["B"]), onehot_column(feat["X"])]
    onehot = (feat["spacegroup"].to_numpy()[:, None] == np.array(sg_levels)[None, :]).astype(float)
    if kind == "spacegroup":
        return onehot
    ang = feat["min_bxb"].to_numpy(float)
    missing = np.isnan(ang)
    # Sin X compartido por dos B (no hay red de vértices): indicador y valor neutro de 180°.
    return np.c_[onehot, feat["is_pm3m"].to_numpy(float), np.where(missing, 180.0, ang), missing]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0, help="semilla 0 de configs/seeds.yaml")
    args = ap.parse_args()
    repo = args.repo.resolve()

    # Etiquetas y orden de índices de sus particiones (filas de id_prop.csv ordenadas por id
    # como texto, el mismo orden que `1PU_Split.py`; ver s2_splits.py).
    with open(repo / "data" / "id_prop.csv") as f:
        rows = sorted(csv.reader(f), key=lambda r: r[0])
    ids = [r[0] for r in rows]
    label = pd.Series([int(r[1]) for r in rows], index=ids)
    with open(repo / "splits_Perov_All.json") as f:
        splits = json.load(f)
    feat = pd.read_csv(args.features, dtype={"id": str}).set_index("id").loc[ids]
    n_err = int(feat["error"].notna().sum())
    feat = feat.assign(spacegroup=feat["spacegroup"].fillna(0).astype(int),
                       is_pm3m=feat["is_pm3m"].fillna(0))  # fmt: skip
    # Elementos de los sitios A, B y X de cada cristal (`Perov_All_ABC.json`).
    with open(repo / "data" / "Perov_All_ABC.json") as f:
        abc = json.load(f)
    feat = feat.assign(A=[abc[i][0] for i in ids], B=[abc[i][1] for i in ids],
                       X=[abc[i][2] for i in ids])  # fmt: skip

    # Su ensemble: [ids, media, etiqueta, DE] en `predict/Perov_All.json`.
    with open(repo / "predict" / "Perov_All.json") as f:
        pid, pmean, _, _ = json.load(f)
    gu = pd.Series(np.asarray(pmean, float), index=[str(i) for i in pid]).loc[ids]

    p_test = sorted({ids[i] for i in splits[0]["PTest"]})
    assert all(sorted(ids[i] for i in s["PTest"]) == p_test for s in splits)  # mismo PTest
    u_all = label.index[label == 0]

    # Descriptivo por clase.
    def describe(sel: pd.Index) -> dict:
        f = feat.loc[sel]
        return {"n": len(sel), "frac_pm3m": round(float(f["is_pm3m"].mean()), 4),
                "frac_cubic": round(float((f["crystal_system"] == "cubic").mean()), 4),
                "median_min_bxb": round(float(f["min_bxb"].median()), 2),
                "frac_bxb_ge_179": round(float((f["min_bxb"] >= 179).mean()), 4),
                "frac_no_bxb": round(float(f["min_bxb"].isna().mean()), 4)}  # fmt: skip

    desc = {"P (todas)": describe(label.index[label == 1]), "PTest": describe(pd.Index(p_test)),
            "U (todas)": describe(u_all)}  # fmt: skip
    # Fracción con X = O o F por clase (química del anión, sin estructura).
    for key, sel in (("P (todas)", label.index[label == 1]), ("PTest", pd.Index(p_test)),
                     ("U (todas)", u_all)):  # fmt: skip
        desc[key]["frac_x_O_or_F"] = round(float(feat.loc[sel, "X"].isin(["O", "F"]).mean()), 4)

    # Modelos de solo simetría con su protocolo de 100 particiones.
    sg_levels = sorted(feat["spacegroup"].unique())
    scores = {"gu_ensemble": gu, "rule_not_pm3m": 1.0 - feat["is_pm3m"].astype(float)}
    rng = np.random.SeedSequence(args.seed)
    model_seeds = rng.generate_state(len(splits))
    models = (("lr_spacegroup", "spacegroup"), ("rf_symmetry_angle", "angle"),
              ("lr_x_element", "x_element"), ("lr_abx_elements", "abx_elements"))  # fmt: skip
    for name, kind in models:
        X = design(feat, kind, sg_levels)
        acc = np.zeros(len(ids))
        for k, s in enumerate(splits):
            tr_p = s["PTrain"] + s["PVal"]
            tr_n = s["NTrain"] + s["NVal"]
            idx = np.r_[tr_p, tr_n]
            y = np.r_[np.ones(len(tr_p)), np.zeros(len(tr_n))]
            if kind != "angle":
                m = LogisticRegression(C=1.0, max_iter=2000)
            else:
                m = RandomForestClassifier(n_estimators=100, min_samples_leaf=2, n_jobs=1,
                                           random_state=int(model_seeds[k]))  # fmt: skip
            m.fit(X[idx], y)
            acc += m.predict_proba(X)[:, 1]
        scores[name] = pd.Series(acc / len(splits), index=ids)

    # Punto de operación de su ensemble (umbral 0,5): P(ŷ=1|U) sobre todas las U.
    gu_ppr_u = float((gu.loc[u_all] >= 0.5).mean())
    gu_tpr05 = float((gu.loc[p_test] >= 0.5).mean())
    table = []
    for name, sc in scores.items():
        sp, su = sc.loc[p_test].to_numpy(), sc.loc[u_all].to_numpy()
        y_true = np.r_[np.ones(len(sp)), np.zeros(len(su))]
        row = {
            "model": name,
            "auc_ptest_vs_u_lower_bound": round(roc_auc_score(y_true, np.r_[sp, su]), 4),
            f"tpr_at_gu_ppr_u_{gu_ppr_u:.3f}": round(tpr_at_ppr_u(sp, su, gu_ppr_u), 4),
        }
        for q in FIXED_PPR_U:
            row[f"tpr_at_ppr_u_{q:.2f}"] = round(tpr_at_ppr_u(sp, su, q), 4)
        table.append(row)
    res = pd.DataFrame(table)

    args.out.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out / "provenance_summary.csv", index=False)
    meta = {"n_crystals": len(ids), "n_feature_errors": n_err, "n_ptest": len(p_test),
            "n_u": len(u_all), "gu_threshold_0p5": {"tpr_ptest": round(gu_tpr05, 4),
            "ppr_u": round(gu_ppr_u, 4)}, "descriptive": desc}  # fmt: skip
    (args.out / "provenance_summary.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", "utf-8"
    )
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(res.to_string(index=False))


if __name__ == "__main__":  # requisito en Windows (regla 2.2)
    main()
