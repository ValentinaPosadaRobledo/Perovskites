"""Informe de F3.5, S2 (preregistro §10b): efecto de la fuga entre polimorfos en Gu et al. (2022).

Variante (a): sus particiones por cristal. Variante (b): las mismas particiones y el MISMO test,
sin ningún cristal de train/val cuya composición esté en el test. Ambas se reentrenan en el
entorno aislado de F3.5 (`scripts/train_split.py` allí) y sus puntuaciones de test, alineadas
por cristal, se copian a `results/f3_5_gu/S2_a/` y `S2_b/` (`test_xxx.json`; solo números
derivados de los datos públicos de Gu, sin estructuras).

Medidas (positivas de test, el mismo conjunto en las 100 particiones):
- h(c) = fracción de modelos que predicen positiva la positiva c (umbral 0,5 de Gu), como h(c)
  sobre semillas en el preregistro §8; TPR = media de h(c). También la TPR del ensemble
  (media de puntuaciones > 0,5), la métrica que publica Gu.
- ΔTPR(a − b) pareada por cristal, con IC bootstrap jerárquico por composición (nivel 1,
  composiciones; nivel 2, cristales), B = 10 000, semilla derivada de `bootstrap_seed`.
- Por separado, las positivas de test CON un polimorfo en train/val (donde puede haber fuga) y
  SIN él (control: Δ ≈ 0).

Uso:
    uv run python scripts/report_f3_5_s2.py
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pandas as pd

from perovskite_pu.evaluation.stats import hierarchical_bootstrap
from perovskite_pu.utils.paths import RAW_DIR, RESULTS_DIR
from perovskite_pu.utils.run_metadata import git_commit, git_is_dirty
from perovskite_pu.utils.seeds import load_seed_config

OUT = RESULTS_DIR / "f3_5_gu"
GU = (
    RAW_DIR
    / "gu2022"
    / "extracted"
    / "kaist-amsg-PerovskiteSynthesizability_Manuscript2021-e12089a"  # pragma: allowlist secret
)
# Clave fija del flujo del bootstrap de S2 dentro de `bootstrap_seed` (distinta de las de H1–H3).
S2_SPAWN_KEY = (100,)


def composition_of() -> dict[str, str]:
    """Composición reducida {A, B, X₃} de cada cristal (todos son ABX₃)."""
    abc = json.loads((GU / "data" / "Perov_All_ABC.json").read_text("utf-8"))
    return {cid: "-".join(sorted((a, b))) + f"|{x}3" for cid, (a, b, x) in abc.items()}


def load_variant(variant: str) -> pd.DataFrame:
    """Una fila por (modelo, cristal de test) con su puntuación y etiqueta."""
    rows = []
    for p in sorted((OUT / f"S2_{variant}").glob("test_*.json")):
        rec = json.loads(p.read_text("utf-8"))
        for cid, score in rec["scores"].items():
            rows.append({"split": rec["split"], "crystal": cid, "score": score,
                         "label": int(rec["labels"][cid])})  # fmt: skip
    return pd.DataFrame(rows)


def leaked_positives(comp: dict[str, str]) -> set[str]:
    """Positivas de test con un polimorfo (misma composición) en train/val de la partición 0."""
    with open(GU / "data" / "id_prop.csv") as f:
        rows = sorted(csv.reader(f), key=lambda r: r[0])  # orden de índices de 1PU_Split.py
    sp = json.loads((GU / "splits_Perov_All.json").read_text("utf-8"))[0]
    seen = {comp[rows[i][0]] for k in ("PTrain", "PVal", "NTrain", "NVal") for i in sp[k]}
    return {rows[i][0] for i in sp["PTest"] if comp[rows[i][0]] in seen}


def main() -> None:
    comp = composition_of()
    a, b = load_variant("a"), load_variant("b")
    common = sorted(set(a["split"]) & set(b["split"]))
    if not common:
        raise SystemExit("Sin modelos terminados en las dos variantes.")
    a, b = a[a["split"].isin(common)], b[b["split"].isin(common)]
    pos_a, pos_b = a[a["label"] == 1], b[b["label"] == 1]
    h_a = pos_a.assign(hit=pos_a["score"] > 0.5).groupby("crystal")["hit"].mean()
    h_b = pos_b.assign(hit=pos_b["score"] > 0.5).groupby("crystal")["hit"].mean()
    ens_a = pos_a.groupby("crystal")["score"].mean() > 0.5
    ens_b = pos_b.groupby("crystal")["score"].mean() > 0.5
    units = pd.Series({c: comp[c] for c in h_a.index})
    leaked = leaked_positives(comp)
    seed = load_seed_config().bootstrap_seed
    rows = []
    for name, sel in (("todas", h_a.index), ("con polimorfo en train/val", h_a.index.intersection(sorted(leaked))),
                      ("sin polimorfo en train/val", h_a.index.difference(sorted(leaked)))):  # fmt: skip
        d = (h_a[sel] - h_b[sel]).astype(float)
        rng = np.random.default_rng(
            np.random.SeedSequence(seed, spawn_key=(*S2_SPAWN_KEY, len(rows)))
        )
        r = hierarchical_bootstrap(d, units[sel], rng=rng) if len(d) else None
        rows.append({"positivas de test": name, "n": len(d),
                     "TPR (a) por cristal": h_a[sel].mean(), "TPR (b) agrupada": h_b[sel].mean(),
                     "ΔTPR (a − b)": d.mean(),
                     "IC 95 %": f"[{r.ci_low:+.3f}, {r.ci_high:+.3f}]" if r else "—",
                     "TPR ensemble (a)": ens_a[sel].mean(), "TPR ensemble (b)": ens_b[sel].mean()})  # fmt: skip
    table = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "s2_summary.csv", index=False, lineterminator="\n")
    u_a = a[a["label"] == 0].groupby("crystal")["score"].mean() > 0.5
    u_b = b[b["label"] == 0].groupby("crystal")["score"].mean() > 0.5
    md = f"""# Informe F3.5, S2: fuga entre polimorfos en la partición de Gu et al. (2022)

> Generado por `scripts/report_f3_5_s2.py` (commit `{git_commit()[:7]}`{", árbol sucio" if git_is_dirty() else ""}). No editar a mano.
> Modelos terminados en las dos variantes: {len(common)} de 100. Secundario (§10b), no confirmatorio.

## Diseño

- **(a)** sus 100 particiones por cristal; **(b)** las mismas, con el mismo test, sin ningún
  cristal de train/val cuya composición esté en el test. Ambas reentrenadas en el entorno
  aislado (CPU, mismo código, semilla 0 de `configs/seeds.yaml`).
- h(c) = fracción de modelos que predicen positiva la positiva c (umbral 0,5); ΔTPR pareada
  por cristal, IC bootstrap jerárquico por composición (B = 10 000).
- {len(leaked)} de las {len(h_a)} positivas de test tienen un polimorfo en train/val (partición 0):
  la fuga solo puede actuar en ellas; las demás son el control.

## Resultados

{table.to_markdown(index=False, floatfmt=".3f")}

P(ŷ=1) del ensemble en las no etiquetadas de test: (a) {u_a.mean():.3f}, (b) {u_b.mean():.3f}.
"""
    (OUT / "informe_s2.md").write_text(md, "utf-8")
    print(md)


if __name__ == "__main__":  # requisito en Windows (regla 2.2)
    main()
