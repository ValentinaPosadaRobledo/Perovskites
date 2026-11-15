"""Informe de F3.6: análisis de sensibilidad frente al análisis principal (F3.3), pareado.

Para cada sensibilidad con corrida en `results/runs_index.csv` (la última de cada experimento
`f3_6_*`), compara su modelo con el de referencia **en las mismas composiciones** y de forma
pareada por composición (h(c), preregistro §8). La referencia es el `pu_pre_dft` de F3.3, salvo
en la sensibilidad 4 (r2SCAN frente a GGA dentro de la misma corrida). En las sensibilidades de
exclusión los folds y semillas son los de F3.3, así que la diferencia se debe a la exclusión;
en las de particiones nuevas (6, 12, 13) el emparejamiento es por composición dentro de cada
esquema. Descriptivo: el §10 no preregistra pruebas para las sensibilidades.

Lee los `raw/` de las corridas (locales, no versionados) y τ de F2.4 (para H2). Escribe
`results/f3_6_sensibilidad/informe.md` y `paired.csv`.

Uso:
    uv run python scripts/report_f3_6.py
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

from perovskite_pu.evaluation.stats import composition_hits, matched_ppr_hits
from perovskite_pu.utils.paths import RESULTS_DIR, ROOT
from perovskite_pu.utils.run_metadata import git_commit, git_is_dirty

OUT = RESULTS_DIR / "f3_6_sensibilidad"
REF_EXP, TAU_EXP, REF_MODEL = "f3_3_pu", "f2_4_baselines", "pu_pre_dft"
SCHEMES = ["random", "group_system", "loeo_A", "loeo_B", "family", "cluster", "temporal"]
CONFIRMATORY = ["loeo_A", "loeo_B", "family", "cluster"]


@dataclass(frozen=True)
class Comparison:
    """Una tabla del informe: modelo de la sensibilidad frente a su referencia."""

    label: str
    model: str
    ref_model: str = REF_MODEL
    ref_in_same_run: bool = False  # True: la referencia es otro modelo de la misma corrida
    subset: str | None = None  # "outside_tables1": solo composiciones fuera de la TableS1


# Sensibilidades en el orden del preregistro (§10); solo se informan las que tienen corrida.
SENSITIVITIES: dict[str, tuple[str, list[Comparison]]] = {
    "f3_6_s01_high_pressure": ("1. Sin alta presión ni metaestables",
                               [Comparison("PU pre_dft", REF_MODEL)]),
    "f3_6_s02_ambiguous_oxidation": ("2. Sin estados de oxidación ambiguos",
                                     [Comparison("PU pre_dft", REF_MODEL)]),
    "f3_6_s04_r2scan": ("4. r2SCAN frente a GGA en `post_dft` (173 composiciones; descriptiva)",
                        [Comparison("post_dft r2SCAN frente a GGA", "pu_post_dft_r2scan",
                                    "pu_post_dft", ref_in_same_run=True)]),
    "f3_6_s05_angle140": ("5. Ángulo B–O–B mínimo de 140°",
                          [Comparison("PU pre_dft", REF_MODEL)]),
    "f3_6_s06_loeo_B_unmerged": ("6. `loeo_B` sin fusionar grupos pequeños",
                                 [Comparison("PU pre_dft", REF_MODEL)]),
    "f3_6_s08_recall": ("8. Recall nominal 0,80 y 0,95",
                        [Comparison("recall 0,80", "pu_pre_dft@r080"),
                         Comparison("recall 0,95", "pu_pre_dft@r095"),
                         Comparison("recall 0,90 (control: debe dar Δ = 0)", "pu_pre_dft@r090")]),
    "f3_6_s11_charge_balanced_u": ("11. Solo U con balance de carga",
                                   [Comparison("PU pre_dft", REF_MODEL)]),
    "f3_6_s12_sites_by_size": ("12. Sitios A/B por tamaño en `loeo_A` y `loeo_B`",
                               [Comparison("PU pre_dft", REF_MODEL)]),
    "f3_6_s13_ambiguous_as_positive": ("13. Ambiguas como positivas (folds nuevos)",
                                       [Comparison("PU pre_dft", REF_MODEL)]),
    "f3_6_s14_pre_dft_tau": ("14. `pre_dft_tau` (+ τ)",
                             [Comparison("todas (τ in-sample en la TableS1)", "pu_pre_dft_tau"),
                              Comparison("solo fuera de la TableS1", "pu_pre_dft_tau",
                                         subset="outside_tables1")]),
}  # fmt: skip


def last_run(index: pd.DataFrame, exp: str) -> pd.Series | None:
    """Última corrida de un experimento en `runs_index.csv` (None si no hay)."""
    rows = index[index["experiment"] == exp]
    return rows.iloc[-1] if len(rows) else None


def load_raw(run: pd.Series) -> pd.DataFrame:
    """Todas las filas `raw/seed_n.csv` de una corrida."""
    files = sorted((ROOT / run["run_dir"] / "raw").glob("seed_*.csv"))
    return pd.concat([pd.read_csv(p) for p in files], ignore_index=True)


def ppr_u(raw: pd.DataFrame, model: str, scheme: str, keep: set[str]) -> float:
    """P(ŷ=1|U) micro-agregada (media de las semillas) en las U de `keep`."""
    d = raw[(raw["baseline"] == model) & (raw["scheme"] == scheme) & (raw["label"] == 0)
            & raw["composition"].isin(keep)]  # fmt: skip
    return float(d.groupby("seed")["pred"].mean().mean()) if len(d) else float("nan")


def paired(sens: pd.DataFrame, model: str, ref: pd.DataFrame, ref_model: str,
           tau: pd.DataFrame, keep: Iterable[str]) -> pd.DataFrame:  # fmt: skip
    """Por esquema: TPR, P(ŷ=1|U), caída de H1 y Δ de H2, en las mismas composiciones `keep`."""
    keep = set(keep)
    s = sens[sens["composition"].isin(keep)]
    r = ref[ref["composition"].isin(keep)]
    t = tau[(tau["baseline"] == "tau_bartel") & tau["composition"].isin(keep)]
    rows = []
    for sch in SCHEMES:
        if sch not in set(s.loc[s["baseline"] == model, "scheme"]) or sch not in set(r["scheme"]):
            continue
        h_s, h_r = composition_hits(s, model, sch), composition_hits(r, ref_model, sch)
        common = h_s.index.intersection(h_r.index)
        row = {"scheme": sch, "n_pos": len(common),
               "tpr_sens": float(h_s[common].mean()), "tpr_ref_same": float(h_r[common].mean()),
               "delta_tpr": float((h_s[common] - h_r[common]).mean()),
               "ppr_u_sens": ppr_u(s, model, sch, keep),
               "ppr_u_ref_same": ppr_u(r, ref_model, sch, keep)}  # fmt: skip
        # H2 (descriptivo): cada modelo a la P(ŷ=1) de τ en cada fold, composiciones con τ.
        # Solo si los folds de τ (F2.4, congelados) son los del modelo: no en particiones nuevas.
        for tag, data, m in (("sens", s, model), ("ref", r, ref_model)):
            try:
                mh = matched_ppr_hits(data[data["baseline"] == m], t, sch)
                row[f"h2_delta_{tag}"] = float((mh["h_model"] - mh["h_ref"]).mean())
            except ValueError:  # folds distintos de los de τ (sensibilidades 6, 12, 13)
                row[f"h2_delta_{tag}"] = float("nan")
        rows.append(row)
    out = pd.DataFrame(rows)
    if "random" in set(out["scheme"]):
        base = out.set_index("scheme")
        for tag in ("sens", "ref_same"):
            out[f"h1_drop_{tag}"] = base.at["random", f"tpr_{tag}"] - out[f"tpr_{tag}"]
    return out


def table(df: pd.DataFrame) -> str:
    """Tabla en markdown con nombres en español y 3 decimales."""
    show = df.drop(columns=["sensitivity", "comparison"], errors="ignore").rename(columns={
        "scheme": "esquema", "n_pos": "P", "tpr_sens": "TPR sens.", "tpr_ref_same": "TPR ref.",
        "delta_tpr": "Δ TPR", "ppr_u_sens": "P(ŷ=1|U) sens.", "ppr_u_ref_same": "P(ŷ=1|U) ref.",
        "h1_drop_sens": "caída H1 sens.", "h1_drop_ref_same": "caída H1 ref.",
        "h2_delta_sens": "Δ H2 sens.", "h2_delta_ref": "Δ H2 ref."})  # fmt: skip
    return show.to_markdown(index=False, floatfmt="+.3f").replace("+nan", "—")


def main() -> None:
    index = pd.read_csv(RESULTS_DIR / "runs_index.csv")
    ref_run, tau_run = last_run(index, REF_EXP), last_run(index, TAU_EXP)
    ref, tau = load_raw(ref_run), load_raw(tau_run)
    ref_comps = pd.read_csv(ROOT / ref_run["run_dir"] / "raw" / "compositions.csv")
    outside_s1 = set(ref_comps.loc[~ref_comps["in_bartel2019"].astype(bool), "composition"])
    sections, frames = [], []
    for exp, (title, comparisons) in SENSITIVITIES.items():
        run = last_run(index, exp)
        if run is None:
            sections.append(f"### {title}\n\nSin corrida todavía.\n")
            continue
        meta = json.loads((ROOT / run["run_dir"] / "run_metadata.json").read_text("utf-8"))
        sens = load_raw(run)
        ex = meta.get("excluded", {})
        text = [f"### {title}\n\n`{run['run_dir']}` (commit `{str(run['git_commit'])[:7]}`, "
                f"{int(run['n_seeds'])} semillas, {run['total_s'] / 60:.0f} min en el último "
                f"tramo). Excluidas: {ex.get('positive', 0)} P y {ex.get('unlabeled', 0)} U; "
                f"folds vacíos tras la exclusión: {meta.get('folds_empty_after_exclusion', 0)}.\n"]  # fmt: skip
        added = set()
        if "added_ambiguous" in pd.read_parquet(ROOT / meta["config"]["data"]["input"]).columns:
            pool = pd.read_parquet(ROOT / meta["config"]["data"]["input"])
            added = set(pool.loc[pool["added_ambiguous"], "composition"])
        for comp in comparisons:
            keep = set(sens["composition"]) - added  # las añadidas no tienen referencia
            if comp.subset == "outside_tables1":
                keep &= outside_s1
            ref_data = sens if comp.ref_in_same_run else ref
            p = paired(sens, comp.model, ref_data, comp.ref_model, tau, keep)
            p = p.assign(sensitivity=exp, comparison=comp.label)
            frames.append(p)
            text.append(f"**{comp.label}** (`{comp.model}` frente a "
                        f"`{comp.ref_model}`{' de esta corrida' if comp.ref_in_same_run else ' de F3.3'}):\n\n"
                        f"{table(p)}\n")  # fmt: skip
        if added:
            # Las ambiguas añadidas como P: TPR propia por esquema (no hay referencia en F3.3).
            rows = []
            for sch in sorted(set(sens["scheme"]), key=SCHEMES.index):
                h = composition_hits(sens[sens["composition"].isin(added)], REF_MODEL, sch)
                rows.append({"esquema": sch, "ambiguas añadidas": len(h), "TPR": h.mean()})
            text.append("Ambiguas añadidas como positivas (sin referencia en F3.3):\n\n"
                        + pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3f") + "\n")  # fmt: skip
        sections.append("\n".join(text))
    OUT.mkdir(parents=True, exist_ok=True)
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(OUT / "paired.csv", index=False,
                                                    lineterminator="\n")  # fmt: skip
    body = "\n".join(sections)
    md = f"""# Informe F3.6: análisis de sensibilidad (pareado frente a F3.3)

> Generado por `scripts/report_f3_6.py` (commit `{git_commit()[:7]}`{", árbol sucio" if git_is_dirty() else ""}). No editar a mano.
> Referencia: `{ref_run["run_dir"]}` (F3.3) y τ de `{tau_run["run_dir"]}` (F2.4). Solo `train_pool`.

## Qué se compara

- Mismo protocolo que F3.3. En las sensibilidades de exclusión (1, 2, 5, 11), las composiciones
  excluidas se quitan de train y test de los **mismos folds**, con las mismas semillas por
  tarea; en las de particiones nuevas (6, 12, 13), los folds cambian.
- **Pareado por composición** (preregistro §8): h(c) es la fracción de semillas en que se
  acierta con la positiva c. "TPR ref." es la de la referencia en las MISMAS positivas;
  Δ TPR = media de h_sens(c) − h_ref(c).
- **Caída H1** = TPR en `random` − TPR en el esquema. **Δ H2** = TPR a la P(ŷ=1) de τ menos
  la de τ (fold a fold, composiciones con τ definido); solo donde los folds son los de τ.
- Descriptivo (medias de {len(ref["seed"].unique())} semillas): el §10 no preregistra pruebas
  para las sensibilidades. Esquemas confirmatorios: {", ".join(f"`{s}`" for s in CONFIRMATORY)}.

## Resultados

{body}"""
    (OUT / "informe.md").write_text(md, "utf-8")
    print(md)


if __name__ == "__main__":  # requisito en Windows (regla 2.2)
    main()
