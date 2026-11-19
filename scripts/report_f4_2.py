"""F4.2: pruebas confirmatorias de H1 y H2 sobre la validación cruzada (preregistro §8–9).

Lee los `raw/` por composición de la última corrida completada del modelo PU (F3.3) y de τ
(F2.4), y aplica el bootstrap jerárquico pareado de `perovskite_pu.evaluation.stats`:

- **H1:** d(c) = h_random(c) − h_g(c) del modelo PU `pre_dft` en cada esquema confirmatorio;
  criterio: valor p bilateral con Holm < α y ΔTPR ≥ δ₁.
- **H2:** d(c) = h_PU(c) − h_τ(c) a igual P(ŷ=1) fold a fold (§7.2); no inferioridad si
  P*(Δ* ≤ −δ₂) con Holm < 0,025. Análisis principal (con los compuestos de Bartel, donde τ es
  *in-sample*) y sensibilidad sin ellos.
- H1 descriptivo (estimación puntual y DE entre semillas) para los modelos de
  `descriptive_models`.

Todos los parámetros vienen de `configs/experiments/f4_2_stats.yaml`; las semillas, de
`configs/seeds.yaml`. No entrena nada y no lee el conjunto reservado.

Uso:
    uv run python scripts/report_f4_2.py [--config ...] [--pu-run-dir ...] [--tau-run-dir ...]
        [--pu-model ...] [--out-dir ...]

`--pu-run-dir` y `--pu-model` permiten una prueba en seco con otra corrida equivalente (p. ej.
la variante a recall 0,90 de la sensibilidad 8, idéntica a F3.3); el informe lo declara.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml

from perovskite_pu.evaluation.stats import (
    bootstrap_rng,
    bootstrap_units,
    composition_hits,
    h1_decision,
    h2_decision,
    hierarchical_bootstrap,
    matched_ppr_hits,
)
from perovskite_pu.utils.paths import RESULTS_DIR, ROOT
from perovskite_pu.utils.run_metadata import git_commit, git_is_dirty
from perovskite_pu.utils.seeds import load_seed_config


def last_completed_run(index: pd.DataFrame, experiment: str) -> Path:
    """Carpeta de la última corrida de un experimento en `runs_index.csv`."""
    rows = index[index["experiment"] == experiment]
    if rows.empty:
        raise FileNotFoundError(f"No hay corridas de {experiment} en runs_index.csv.")
    return ROOT / rows.iloc[-1]["run_dir"]


def load_raw(run_dir: Path, seeds: list[int]) -> tuple[pd.DataFrame, str]:
    """`raw/seed_n.csv` de las semillas pedidas y un SHA-256 conjunto de esos archivos.

    Raises:
        FileNotFoundError: si falta alguna semilla (no se analiza con semillas incompletas).
    """
    meta = json.loads((run_dir / "run_metadata.json").read_text("utf-8"))
    if meta.get("status", "completed") != "completed":
        raise RuntimeError(f"La corrida {run_dir.name} no está completada.")
    paths = [run_dir / "raw" / f"seed_{s}.csv" for s in seeds]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Faltan semillas en {run_dir}: {missing}")
    digest = hashlib.sha256()
    for p in paths:
        digest.update(p.read_bytes())
    raw = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    return raw, digest.hexdigest()


def family_map(data_cfg: dict) -> dict[str, str]:
    """Elemento B → familia, con la familia por defecto para los no listados."""
    fam = data_cfg["b_families"]
    out = {el: name for name, els in fam.items() if name != "default_family" for el in els}
    return out | {"__default__": fam["default_family"]}


def per_seed_tpr(raw: pd.DataFrame, model: str, scheme: str) -> pd.Series:
    """TPR micro-agregada de cada semilla (para la DE entre semillas)."""
    d = raw[(raw["baseline"] == model) & (raw["scheme"] == scheme) & (raw["label"] == 1)]
    return d.groupby("seed")["pred"].mean()


def fmt_p(p: float, n_boot: int) -> str:
    """Valor p con 4 decimales; un 0 del bootstrap se informa como < 1/B (§9)."""
    return f"< {1 / n_boot:.0e}" if p == 0 else f"{p:.4f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=ROOT / "configs/experiments/f4_2_stats.yaml")
    ap.add_argument("--pu-run-dir", type=Path, default=None)
    ap.add_argument("--tau-run-dir", type=Path, default=None)
    ap.add_argument("--pu-model", default=None, help="nombre del modelo PU en el raw/")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text("utf-8"))
    st, inp = cfg["stats"], cfg["inputs"]
    seed_cfg = load_seed_config()
    seeds = list(seed_cfg.seeds)
    index = pd.read_csv(RESULTS_DIR / "runs_index.csv")
    pu_dir = (args.pu_run_dir or last_completed_run(index, inp["pu_experiment"])).resolve()
    tau_dir = (args.tau_run_dir or last_completed_run(index, inp["tau_experiment"])).resolve()
    pu_model = args.pu_model or inp["pu_model"]
    dry_run = args.pu_run_dir is not None or args.pu_model is not None
    out_dir = args.out_dir or RESULTS_DIR / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    raw, pu_sha = load_raw(pu_dir, seeds)
    tau_raw, tau_sha = load_raw(tau_dir, seeds)
    raw_pu = raw[raw["baseline"] == pu_model]
    raw_tau = tau_raw[tau_raw["baseline"] == inp["tau_model"]]
    if raw_pu.empty or raw_tau.empty:
        raise ValueError(f"Sin filas de {pu_model} o de {inp['tau_model']} en los raw/.")
    pool = pd.read_parquet(ROOT / cfg["data"]["input"]).set_index("composition")[["A", "B"]]
    data_cfg = yaml.safe_load((ROOT / cfg["data"]["config"]).read_text("utf-8"))
    fam = family_map(data_cfg)
    b_families = {b: fam.get(b, fam["__default__"]) for b in pool["B"].unique()}
    comps = pd.read_csv(pu_dir / "raw" / "compositions.csv")
    bartel = set(comps.loc[comps["in_bartel2019"].astype(bool), "composition"])
    schemes = list(st["confirmatory_schemes"])
    ref = st["reference_scheme"]
    n_boot, ci, seed = int(st["n_boot"]), float(st["ci"]), seed_cfg.bootstrap_seed

    # H1: caída de TPR de `random` a cada esquema por grupos, pareada por composición.
    h_ref = composition_hits(raw_pu, pu_model, ref)
    h1_res, h1_sd = {}, {}
    for g in schemes:
        h_g = composition_hits(raw_pu, pu_model, g)
        if set(h_g.index) != set(h_ref.index):
            raise ValueError(f"Positivos distintos en {ref} y {g}.")
        d = (h_ref - h_g.reindex(h_ref.index)).rename("d")
        units, strata = bootstrap_units(pool.loc[d.index], g, b_families)
        h1_res[g] = hierarchical_bootstrap(d, units, strata, margin=0.0, n_boot=n_boot,
                                           rng=bootstrap_rng(seed, "H1", g), ci=ci)  # fmt: skip
        h1_sd[g] = float((per_seed_tpr(raw_pu, pu_model, ref)
                          - per_seed_tpr(raw_pu, pu_model, g)).std(ddof=1))  # fmt: skip
    h1 = h1_decision(h1_res, delta1=float(st["delta1"]), alpha=float(st["alpha_h1"]))
    h1["sd_between_seeds"] = pd.Series(h1_sd)

    # H2: PU frente a τ a igual P(ŷ=1); principal y sin los compuestos de Bartel.
    h2_tables = {}
    for label, hyp, excl in (("principal", "H2", set()),
                             ("sin_bartel2019", "H2_sensitivity", bartel)):  # fmt: skip
        res, tprs = {}, {}
        for g in schemes:
            m = matched_ppr_hits(raw_pu, raw_tau, g, exclude=excl)
            d = (m["h_model"] - m["h_ref"]).rename("d")
            units, strata = bootstrap_units(pool.loc[d.index], g, b_families)
            res[g] = hierarchical_bootstrap(d, units, strata, margin=-float(st["delta2"]),
                                            n_boot=n_boot, rng=bootstrap_rng(seed, hyp, g),
                                            ci=ci)  # fmt: skip
            tprs[g] = (float(m["h_model"].mean()), float(m["h_ref"].mean()))
        t = h2_decision(res, delta2=float(st["delta2"]),
                        alpha_one_sided=float(st["alpha_h2_one_sided"]))  # fmt: skip
        t["tpr_pu"] = pd.Series({g: v[0] for g, v in tprs.items()})
        t["tpr_tau"] = pd.Series({g: v[1] for g, v in tprs.items()})
        h2_tables[label] = t

    # H1 descriptivo para otros modelos (sin decisión, §8).
    desc_rows = []
    for model in [] if dry_run else inp.get("descriptive_models", []):
        r = raw[raw["baseline"] == model]
        hr = composition_hits(r, model, ref)
        for g in schemes:
            hg = composition_hits(r, model, g)
            sd = (per_seed_tpr(r, model, ref) - per_seed_tpr(r, model, g)).std(ddof=1)
            desc_rows.append(
                {
                    "model": model,
                    "scheme": g,
                    "delta_tpr": float(hr.mean() - hg.mean()),
                    "sd_between_seeds": sd,
                }
            )
    desc = pd.DataFrame(desc_rows)

    # Salidas: tablas, metadatos de entrada e informe.
    h1.to_csv(out_dir / "h1.csv", lineterminator="\n")
    for label, t in h2_tables.items():
        t.to_csv(out_dir / f"h2_{label}.csv", lineterminator="\n")
    if not desc.empty:
        desc.to_csv(out_dir / "h1_descriptive.csv", index=False, lineterminator="\n")
    meta = {
        "date_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(), "git_dirty": git_is_dirty(), "dry_run": dry_run,
        "config": cfg, "seeds": seeds, "bootstrap_seed": seed, "pu_model": pu_model,
        "pu_run_dir": pu_dir.relative_to(ROOT).as_posix(), "pu_raw_sha256": pu_sha,
        "tau_run_dir": tau_dir.relative_to(ROOT).as_posix(), "tau_raw_sha256": tau_sha,
    }  # fmt: skip
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False)
                                              + "\n", "utf-8")  # fmt: skip
    write_report(out_dir, h1, h2_tables, desc, meta, n_boot)
    print((out_dir / "informe.md").read_text("utf-8"))


def write_report(out_dir: Path, h1: pd.DataFrame, h2: dict[str, pd.DataFrame],
                 desc: pd.DataFrame, meta: dict, n_boot: int) -> None:  # fmt: skip
    """Informe en Markdown con las decisiones de H1 y H2."""
    st = meta["config"]["stats"]
    lines = ["# F4.2: pruebas confirmatorias de H1 y H2 (validación cruzada)", ""]
    if meta["dry_run"]:
        lines += [f"> **Prueba en seco** con `{meta['pu_model']}` de `{meta['pu_run_dir']}`, "
                  "no con la corrida oficial de F3.3. No se cita como resultado.", ""]  # fmt: skip
    lines += [f"Entradas: `{meta['pu_run_dir']}` (modelo `{meta['pu_model']}`) y "
              f"`{meta['tau_run_dir']}` (τ); semillas {meta['seeds'][0]}–{meta['seeds'][-1]}; "
              f"bootstrap jerárquico B = {n_boot}, semilla {meta['bootstrap_seed']}; "
              f"commit `{meta['git_commit'][:7]}`.", ""]  # fmt: skip
    lines += ["## H1 — caída de TPR de `random` a cada esquema (modelo PU `pre_dft`)", "",
              f"Criterio: p bilateral con Holm < {st['alpha_h1']} y ΔTPR ≥ δ₁ = {st['delta1']}.",
              "", "| Esquema | ΔTPR | IC 95 % | DE entre semillas | p | p (Holm) | Unidades / P "
              "| Cumple |", "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for g, r in h1.iterrows():
        lines.append(f"| `{g}` | {r.estimate:.3f} | [{r.ci_low:.3f}; {r.ci_high:.3f}] | "
                     f"{r.sd_between_seeds:.3f} | {fmt_p(r.p_two_sided, n_boot)} | "
                     f"{fmt_p(r.p_holm, n_boot)} | {int(r.n_units)} / {int(r.n_compositions)} | "
                     f"{'sí' if r.criterion else 'no'} |")  # fmt: skip
    lines += ["", f"**Conclusión global (intersección-unión):** H1 "
              f"{'se sostiene' if bool(h1['global'].iloc[0]) else 'no se sostiene en los 4 esquemas'}"
              f" ({int(h1['criterion'].sum())} de {len(h1)} cumplen el criterio).", ""]  # fmt: skip
    for label, t in h2.items():
        title = ("principal (incluye Bartel, τ *in-sample*)" if label == "principal"
                 else "sensibilidad sin los compuestos de Bartel (τ fuera de muestra)")  # fmt: skip
        lines += [f"## H2 — no inferioridad del PU frente a τ: {title}", "",
                  f"Criterio: P*(Δ* ≤ −δ₂) con Holm < {st['alpha_h2_one_sided']} "
                  f"(δ₂ = {st['delta2']}).", "",
                  "| Esquema | TPR τ | TPR PU (igual P(ŷ=1)) | Δ | IC 95 % | p no inf. (Holm) "
                  "| No inferior | Superior | P |", "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
        for g, r in t.iterrows():
            lines.append(f"| `{g}` | {r.tpr_tau:.3f} | {r.tpr_pu:.3f} | {r.estimate:+.3f} | "
                         f"[{r.ci_low:+.3f}; {r.ci_high:+.3f}] | "
                         f"{fmt_p(r.p_noninf_holm, n_boot)} | "
                         f"{'sí' if r.noninferior else 'no'} | {'sí' if r.superior else 'no'} | "
                         f"{int(r.n_compositions)} |")  # fmt: skip
        verdict = bool(t["global_noninferior"].iloc[0])
        lines += ["", f"**Conclusión global:** no inferioridad "
                  f"{'en los 4 esquemas' if verdict else 'no demostrada en los 4 esquemas'} "
                  f"({int(t['noninferior'].sum())} de {len(t)}). Un resultado no concluyente "
                  "no se interpreta como inferioridad ni como equivalencia (§8.4).", ""]  # fmt: skip
    if not desc.empty:
        lines += [
            "## H1 descriptivo (otros modelos, sin decisión)",
            "",
            "| Modelo | Esquema | ΔTPR | DE entre semillas |",
            "|---|---|---|---|",
        ]
        for _, r in desc.iterrows():
            lines.append(f"| `{r.model}` | `{r.scheme}` | {r.delta_tpr:.3f} | "
                         f"{r.sd_between_seeds:.3f} |")  # fmt: skip
        lines.append("")
    (out_dir / "informe.md").write_text("\n".join(lines), "utf-8")


if __name__ == "__main__":  # requisito en Windows (regla 2.2)
    main()
