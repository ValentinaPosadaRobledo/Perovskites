"""Informe de F3.2: prior de clase π (Elkan–Noto y TIcE) y su uso en la sensibilidad.

Lee la última corrida de `f3_2_prior` en `results/runs_index.csv` (su `summary.csv` y su
`raw/`) y, para la sensibilidad, el `summary.csv` de la última corrida de `f3_3_pu`. Escribe
`results/f3_2_prior/informe.md` y `results/f3_2_prior/precision_scar.csv`. No ejecuta
modelos: primero

    uv run python scripts/run_prior.py configs/experiments/f3_2_prior.yaml

Uso:
    uv run python scripts/report_f3_2.py
"""

from __future__ import annotations

import json

import pandas as pd
import yaml

from perovskite_pu.evaluation.prior import precision_under_scar
from perovskite_pu.utils.paths import RESULTS_DIR, ROOT
from perovskite_pu.utils.run_metadata import git_commit, git_is_dirty

EXP = "f3_2_prior"
PU_EXP = "f3_3_pu"
SCHEMES = ["random", "group_system", "loeo_A", "loeo_B", "family", "cluster", "temporal"]
NAMES = {
    "elkan_noto": "Elkan–Noto (RF, isotónica)",
    "elkan_noto_sigmoid": "Elkan–Noto (RF, sigmoide)",
    "elkan_noto_gb": "Elkan–Noto (GB, isotónica)",
    "tice": "TIcE (rango en 4 partes)",
    "tice_quantile": "TIcE (cuantiles)",
    "pu_pre_dft": "PU pre_dft",
    "pu_post_dft": "PU post_dft",
}
POPULATIONS = {"all": "todo train_pool", "charge_balanced_u": "sin U sin balance de carga"}
ROLES = {"principal": "principal", "sensitivity": "sensibilidad"}


def last_run(index: pd.DataFrame, exp: str) -> pd.Series | None:
    """Última corrida de un experimento en `runs_index.csv` (None si no hay)."""
    rows = index[index["experiment"] == exp]
    return rows.iloc[-1] if len(rows) else None


def fmt(row: pd.Series, col: str, digits: int = 3) -> str:
    """'media [IC 95 %] (mín.–máx.)'."""
    m, lo, hi = row[f"{col}_mean"], row[f"{col}_ci95_low"], row[f"{col}_ci95_high"]
    mn, mx = row[f"{col}_min"], row[f"{col}_max"]
    return f"{m:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}] ({mn:.{digits}f}–{mx:.{digits}f})"


def main() -> None:
    index = pd.read_csv(RESULTS_DIR / "runs_index.csv")
    run = last_run(index, EXP)
    if run is None:
        raise SystemExit("No hay corridas de f3_2_prior en runs_index.csv.")
    run_dir = ROOT / run["run_dir"]
    meta = json.loads((run_dir / "run_metadata.json").read_text("utf-8"))
    cfg = meta["config"]
    summary = pd.read_csv(run_dir / "summary.csv")
    raw = pd.concat([pd.read_csv(p) for p in sorted((run_dir / "raw").glob("seed_*.csv"))],
                    ignore_index=True)  # fmt: skip
    seeds = sorted(int(x) for x in raw["seed"].unique())

    # 1. Tabla principal: c, π_U y positivas ocultas por población y método.
    tables = {}
    for pop in POPULATIONS:
        rows = []
        for method in cfg["methods"]:
            hit = summary[(summary["population"] == pop) & (summary["method"] == method)]
            if not len(hit):
                continue
            h = hit.iloc[0]
            rows.append({"método": NAMES.get(method, method), "papel": ROLES.get(h["role"], h["role"]),
                         "c": fmt(h, "c"), "π_U": fmt(h, "pi_u"),
                         "positivas ocultas": fmt(h, "hidden_positives", 0),
                         "recortes": int(h["n_clipped"])})  # fmt: skip
        tables[pop] = pd.DataFrame(rows).to_markdown(index=False)
    sizes = summary.groupby("population")[["n_labeled", "n_unlabeled"]].first()

    # 2. Diagnósticos: calibración de Elkan–Noto (media de g ≈ n_L / N) y pasadas de TIcE.
    diag = []
    for (pop, method), g in raw.groupby(["population", "method"], sort=True):
        n_l, n_u = int(g["n_labeled"].iloc[0]), int(g["n_unlabeled"].iloc[0])
        r = {"población": pop, "método": NAMES.get(method, method)}
        if "g_mean_all" in g and g["g_mean_all"].notna().any():
            r["media de g (todas)"] = f"{g['g_mean_all'].mean():.3f}"
            r["n_L / N"] = f"{n_l / (n_l + n_u):.3f}"
        if "c_first_pass" in g and g["c_first_pass"].notna().any():
            r["c 1.ª pasada"] = f"{g['c_first_pass'].mean():.3f}"
            r["DE entre folds"] = f"{g['c_fold_sd'].mean():.3f}"
        diag.append(r)

    # 3. Rango de π_U con los dos métodos principales (población principal).
    principal = summary[(summary["population"] == "all") & (summary["role"] == "principal")]
    pi_lo, pi_hi = float(principal["pi_u_min"].min()), float(principal["pi_u_max"].max())

    # 4. Sensibilidad: precisión en el punto de operación de F3.3 bajo SCAR.
    pu_run = last_run(index, PU_EXP)
    prec_md = "Sin corrida completada de F3.3 en `runs_index.csv`: sección pendiente."
    if pu_run is not None:
        pu = pd.read_csv(ROOT / pu_run["run_dir"] / "summary.csv")
        prec_rows = []
        for _, r in pu.iterrows():
            lo = precision_under_scar(r["tpr_mean"], r["ppr_u_mean"], pi_lo, r["n_pos"], r["n_u"])
            hi = precision_under_scar(r["tpr_mean"], r["ppr_u_mean"], pi_hi, r["n_pos"], r["n_u"])
            prec_rows.append({"model": r["baseline"], "scheme": r["scheme"],
                              "tpr": r["tpr_mean"], "ppr_u": r["ppr_u_mean"],
                              "pi_u_low": pi_lo, "pi_u_high": pi_hi,
                              "precision_u_low": lo["precision_u_scar"],
                              "precision_u_high": hi["precision_u_scar"],
                              "precision_low": lo["precision_scar"],
                              "precision_high": hi["precision_scar"]})  # fmt: skip
        prec = pd.DataFrame(prec_rows)
        prec.to_csv(RESULTS_DIR / EXP / "precision_scar.csv", index=False, lineterminator="\n")
        show = []
        # Solo el modelo principal: π se estimó con todo train_pool, que es su población.
        # `pu_post_dft` se evalúa solo en las elegibles de post_dft (otra fracción de ocultas).
        for model in ("pu_pre_dft",):
            for sch in SCHEMES:
                hit = prec[(prec["model"] == model) & (prec["scheme"] == sch)]
                if not len(hit):
                    continue
                h = hit.iloc[0]
                show.append({"modelo": NAMES[model], "esquema": sch, "TPR": f"{h['tpr']:.3f}",
                             "P(ŷ=1|U)": f"{h['ppr_u']:.3f}",
                             "precisión entre U predichas": f"{h['precision_u_low']:.2f}–"
                                                            f"{h['precision_u_high']:.2f}",
                             "precisión global": f"{h['precision_low']:.2f}–"
                                                 f"{h['precision_high']:.2f}"})  # fmt: skip
        prec_md = (
            f"Con π_U entre {pi_lo:.3f} y {pi_hi:.3f} (mín. y máx. de los dos métodos principales "
            f"en las {len(seeds)} semillas) y la TPR y P(ŷ=1|U) medias de F3.3 "
            f"(`{pu_run['run_dir']}`, {int(pu_run['n_seeds'])} semillas):\n\n"
            + pd.DataFrame(show).to_markdown(index=False)
            + "\n\nSolo `pu_pre_dft`: π_U se estimó en todo `train_pool`, que es su población de "
            "evaluación. `pu_post_dft` se evalúa en las elegibles de post_dft, con otra fracción "
            "de ocultas. En `temporal`, el test solo contiene las positivas posteriores al corte, "
            "así que la aproximación es más burda."
        )

    md = f"""# Informe F3.2: prior de clase π

> Generado por `scripts/report_f3_2.py` (commit `{git_commit()[:7]}`{", árbol sucio" if git_is_dirty() else ""}) a partir de
> `{run["run_dir"]}` (corrida con el commit `{str(run["git_commit"])[:7]}`{", árbol sucio" if bool(run["git_dirty"]) else ""}; semillas {seeds}; {run["total_s"]:.0f} s).
> No editar a mano. Configuración: `configs/experiments/{EXP}.yaml`. Solo `train_pool`.

## 1. Qué se mide

- **c** = P(s = 1 | y = 1), la fracción de perovskitas sintetizables que ya están
  etiquetadas. De ella, bajo SCAR: α = n_L / (c · N) y **π_U** = n_L (1 − c) / (c · n_U), la
  fracción de perovskitas ocultas entre las no etiquetadas.
- **Elkan–Noto:** media de g(x) ≈ P(s = 1 | x) sobre las positivas, cada una puntuada por un
  clasificador calibrado que no la vio (validación cruzada de {cfg["methods"]["elkan_noto"]["n_splits"]} folds).
- **TIcE** (Bekker y Davis, 2018, Algorithm 1): mayor cota inferior de c en las subregiones
  de un árbol inducido con un fold y evaluado con los demás; valores por defecto del artículo.
- Features `{cfg["features"]["family"]}`. Media de {len(seeds)} semillas, IC 95 % (t de Student) y, entre
  paréntesis, mínimo–máximo. La variación entre semillas solo refleja folds y modelos: el
  conjunto de datos es siempre el mismo.

## 2. Resultados

**{POPULATIONS["all"]}** (n_L = {int(sizes.loc["all", "n_labeled"])}, n_U = {int(sizes.loc["all", "n_unlabeled"])}; c ≥ n_L / N = {sizes.loc["all", "n_labeled"] / sizes.loc["all"].sum():.3f} por construcción):

{tables["all"]}

**{POPULATIONS["charge_balanced_u"]}** (sensibilidad 11; n_L = {int(sizes.loc["charge_balanced_u", "n_labeled"])}, n_U = {int(sizes.loc["charge_balanced_u", "n_unlabeled"])}):

{tables["charge_balanced_u"]}

## 3. Diagnósticos

Si el clasificador de Elkan–Noto está calibrado, la media de g en todas las filas debe ser
≈ n_L / N. En TIcE, la primera pasada usa la cota del peor caso (ĉ = 0,5) y la segunda, la
estimación de la primera.

{pd.DataFrame(diag).fillna("—").to_markdown(index=False)}

## 4. Sensibilidad: precisión en el punto de operación (bajo SCAR)

Si las perovskitas ocultas se detectaran con la misma TPR que las etiquetadas (SCAR), las
positivas ocultas predichas serían TPR · π_U · n_U. Precisión entre las U predichas =
TPR · π_U / P(ŷ=1|U); precisión global, incluyendo las P del test. Es una aproximación de
sensibilidad, no una métrica del protocolo.

{prec_md}

## 5. Lectura (descriptiva)

- **SCAR no se cumple** y el sesgo tiene dirección: las perovskitas conocidas se concentran
  en las químicas más estudiadas, donde la probabilidad de estar etiquetada es mayor. Ambos
  métodos tienden entonces a **sobrestimar c y subestimar π**: TIcE porque toma la región con
  la mayor fracción de etiquetadas, y Elkan–Noto porque promedia g sobre las etiquetadas, que
  provienen sobre todo de esas regiones. Las π_U de este informe se leen como orientativas y
  probablemente bajas.
- La diferencia entre métodos es parte del resultado: π no está identificada sin supuestos
  fuertes, y por eso el protocolo principal no depende de ella (punto de operación por recall).
"""
    out = RESULTS_DIR / EXP / "informe.md"
    out.write_text(md, "utf-8")
    (RESULTS_DIR / EXP / "config_used.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), "utf-8"
    )
    print(md)


if __name__ == "__main__":  # requisito en Windows (regla 2.2)
    main()
