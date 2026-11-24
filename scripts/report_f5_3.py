"""F5.3: lixiviación de los sitios A y B en la ventana fisiológica (tesis, OE4 exploratorio).

Lee la última corrida de F5.1 (`pourbaix.csv`, con `stable_in_window`) y la última de F5.2
(`candidates.csv` de la condición principal, con el frente y las marcas de toxicidad), y
escribe en `results/f5_3_lixiviacion/`:

- `leaching.csv`: una fila por composición × pH × concentración, con los metales disueltos
  (A, B), sus especies y los productos sólidos;
- `summary.csv`: fracción de cada patrón (solo A, solo B, ambos, ninguno) por grupo (P, U, U
  elegibles en F5.2, frente principal de F5.2) y condición;
- `front.csv`: el detalle de las candidatas del frente en todas las condiciones;
- `run_metadata.json` e `informe.md`.

Determinista; no entrena nada y solo lee composiciones de `train_pool`.

Uso:
    uv run python scripts/report_f5_3.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pandas as pd

from perovskite_pu.pourbaix.leaching import leaching_pattern, leaching_row
from perovskite_pu.utils.hashing import sha256_file
from perovskite_pu.utils.paths import RESULTS_DIR, ROOT
from perovskite_pu.utils.run_metadata import git_commit, git_is_dirty

OUT = RESULTS_DIR / "f5_3_lixiviacion"
PATTERNS = ["only_A", "only_B", "both", "none"]


def last_run(index: pd.DataFrame, experiment: str):
    """Carpeta de la última corrida de un experimento en `runs_index.csv`."""
    rows = index[index["experiment"] == experiment]
    if rows.empty:
        raise FileNotFoundError(f"No hay corridas de {experiment} en runs_index.csv.")
    return ROOT / rows.iloc[-1]["run_dir"]


def main() -> None:
    index = pd.read_csv(RESULTS_DIR / "runs_index.csv")
    pbx_path = last_run(index, "f5_1_pourbaix") / "pourbaix.csv"
    cand_path = last_run(index, "f5_2_pareto") / "candidates.csv"
    pbx = pd.read_csv(pbx_path)
    cand = pd.read_csv(cand_path).set_index("composition")

    rows = [{**r, **leaching_row(r["stable_in_window"], r["A"], r["B"])}
            for r in pbx.to_dict("records")]  # fmt: skip
    leach = pd.DataFrame(rows)
    leach["pattern"] = [leaching_pattern(a, b) for a, b in
                        zip(leach["A_dissolved"], leach["B_dissolved"], strict=True)]  # fmt: skip
    # Grupos: P y U de F5.1, y las U elegibles y el frente principal de F5.2.
    eligible = set(cand.index[cand["pareto_rank"] > 0])
    front = set(cand.index[cand["on_front"].astype(bool)])
    groups = {"P": leach["label"] == 1, "U": leach["label"] == 0,
              "U_eligible_F5.2": leach["composition"].isin(eligible),
              "front_F5.2": leach["composition"].isin(front)}  # fmt: skip
    summ = []
    for name, mask in groups.items():
        for (ph, conc), g in leach[mask].groupby(["pH", "ion_conc_M"]):
            freq = g["pattern"].value_counts(normalize=True)
            summ.append({"group": name, "pH": ph, "ion_conc_M": conc, "n": len(g),
                         **{f"frac_{p}": float(freq.get(p, 0.0)) for p in PATTERNS}})  # fmt: skip
    summary = pd.DataFrame(summ)

    OUT.mkdir(parents=True, exist_ok=True)
    lt = {"lineterminator": "\n", "index": False}
    cols = ["composition", "label", "A", "B", "pH", "ion_conc_M", "dg_max", "pattern",
            "A_species", "B_species", "solid_products"]  # fmt: skip
    leach[cols].sort_values(["composition", "pH", "ion_conc_M"]).to_csv(OUT / "leaching.csv", **lt)
    summary.to_csv(OUT / "summary.csv", **lt)
    fr = leach[leach["composition"].isin(front)][cols].sort_values(["composition", "pH",
                                                                    "ion_conc_M"])  # fmt: skip
    fr.to_csv(OUT / "front.csv", **lt)
    meta = {"date_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "git_commit": git_commit(), "git_dirty": git_is_dirty(),
            "inputs": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha256_file(p)}
                       for p in (pbx_path, cand_path)]}  # fmt: skip
    (OUT / "run_metadata.json").write_text(json.dumps(meta, indent=2) + "\n", "utf-8")
    write_report(summary, fr, meta)
    print((OUT / "informe.md").read_text("utf-8"))


def write_report(summary: pd.DataFrame, front: pd.DataFrame, meta: dict) -> None:
    """Informe en Markdown: patrones por grupo y detalle del frente a pH 7,4 y 1e-6 M."""
    main = summary[(summary["pH"] == 7.4) & (summary["ion_conc_M"] == 1e-6)]
    lines = ["# F5.3: lixiviación de los sitios A y B (tesis, exploratorio)", "",
             f"Entradas: {', '.join(f'`{i['path']}`' for i in meta['inputs'])}; commit "
             f"`{meta['git_commit'][:7]}`. Lectura termodinámica del diagrama de Pourbaix: un "
             "sitio se lixivia si su metal aparece en alguna especie disuelta estable en la "
             "ventana de −0,2 a +0,4 V. No es una cinética de disolución.", "",
             "## Patrones a pH 7,4 y 1e-6 M", "",
             "| Grupo | n | Solo A | Solo B | Ambos | Ninguno |", "|---|---|---|---|---|---|"]  # fmt: skip
    for _, r in main.iterrows():
        lines.append(f"| {r.group} | {r.n} | {r.frac_only_A:.0%} | {r.frac_only_B:.0%} | "
                     f"{r.frac_both:.0%} | {r.frac_none:.0%} |")  # fmt: skip
    lines += ["", "## Frente principal de F5.2 a pH 7,4 y 1e-6 M", "",
              "| Composición | ΔG_pbx | Patrón | Especies de A | Especies de B | Sólidos |",
              "|---|---|---|---|---|---|"]  # fmt: skip
    f = front[(front["pH"] == 7.4) & (front["ion_conc_M"] == 1e-6)]
    for _, r in f.iterrows():
        lines.append(f"| {r.composition} | {r.dg_max:.3f} | {r.pattern} | {r.A_species or '—'} "
                     f"| {r.B_species or '—'} | {r.solid_products or '—'} |")  # fmt: skip
    lines.append("")
    (OUT / "informe.md").write_text("\n".join(lines), "utf-8")


if __name__ == "__main__":  # requisito en Windows (regla 2.2)
    main()
