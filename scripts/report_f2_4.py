"""Informe de F2.4: líneas base con todos los esquemas y semillas.

Lee la última corrida de `f2_4_baselines` en `results/runs_index.csv` y su `summary.csv`.
Escribe `results/f2_4_baselines/informe.md`. No ejecuta nada: primero

    uv run poe experiment configs/experiments/f2_4_baselines.yaml

Uso:
    uv run python scripts/report_f2_4.py
"""

from __future__ import annotations

import json

import pandas as pd
import yaml

from perovskite_pu.utils.paths import RESULTS_DIR, ROOT
from perovskite_pu.utils.run_metadata import git_commit, git_is_dirty

EXP = "f2_4_baselines"
SCHEMES = ["random", "group_system", "loeo_A", "loeo_B", "family", "cluster", "temporal"]
CONFIRMATORY = ["loeo_A", "loeo_B", "family", "cluster"]
NAMES = {
    "tau_bartel": "τ < 4,18 (Bartel)",
    "t_window_bartel": "t en [0,825; 1,059] (Bartel)",
    "t_window_classic": "t en [0,8; 1,0] (clásica)",
    "ehull_0p05": "E_hull ≤ 0,05",
    "ehull_0": "E_hull = 0",
    "ehull_0p1": "E_hull ≤ 0,1",
    "ehull_model": "Modelo solo E_hull",
    "symmetry_model": "Modelo solo simetría",
    "charge_balance_icsd": "Balance de carga (ICSD)",
    "charge_balance_common": "Balance de carga (comunes)",
    "trivial_random": "Azar",
}


def cell(row: pd.Series, metric: str) -> str:
    """'media [IC 95 %]' con 3 decimales."""
    m, lo, hi = row[f"{metric}_mean"], row[f"{metric}_ci95_low"], row[f"{metric}_ci95_high"]
    if pd.isna(m):
        return "—"
    return f"{m:.3f}" if abs(hi - lo) < 5e-4 else f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def table(summary: pd.DataFrame, metric: str, baselines: list[str]) -> str:
    """Tabla línea base × esquema de una métrica."""
    rows = []
    for b in baselines:
        r = {"línea base": NAMES.get(b, b)}
        for sch in SCHEMES:
            hit = summary[(summary["baseline"] == b) & (summary["scheme"] == sch)]
            r[sch] = cell(hit.iloc[0], metric) if len(hit) else "—"
        rows.append(r)
    return pd.DataFrame(rows).to_markdown(index=False)


def main() -> None:
    index = pd.read_csv(RESULTS_DIR / "runs_index.csv")
    run = index[index["experiment"] == EXP].iloc[-1]
    run_dir = ROOT / run["run_dir"]
    summary = pd.read_csv(run_dir / "summary.csv")
    meta = json.loads((run_dir / "run_metadata.json").read_text("utf-8"))
    cfg = meta["config"]
    roles = {b: spec["role"] for b, spec in cfg["baselines"].items()}
    principal = [b for b in NAMES if roles.get(b) == "principal"]
    sensitivity = [b for b in NAMES if roles.get(b) == "sensitivity"]

    sizes = summary[summary["scheme"] == "random"].set_index("baseline")[["n_pos", "n_u"]]
    size_rows = [{"línea base": NAMES[b], "subconjunto": cfg["baselines"][b]["subset"],
                  "P": int(sizes.loc[b, "n_pos"]), "U": int(sizes.loc[b, "n_u"])}
                 for b in NAMES if b in sizes.index]  # fmt: skip

    def val(b: str, sch: str, metric: str) -> float:
        hit = summary[(summary["baseline"] == b) & (summary["scheme"] == sch)]
        return float(hit.iloc[0][f"{metric}_mean"])

    sym_r, sym_u = val("symmetry_model", "random", "tpr"), val("symmetry_model", "random", "ppr_u")
    eh_r, eh_u = val("ehull_model", "random", "tpr"), val("ehull_model", "random", "ppr_u")
    tau_r, tau_u = val("tau_bartel", "random", "tpr"), val("tau_bartel", "random", "ppr_u")
    learned_drop = {
        b: max(val(b, "random", "tpr") - val(b, s, "tpr") for s in CONFIRMATORY)
        for b in ("ehull_model", "symmetry_model")
    }
    fixed = [f"{x:.2f}" for x in cfg["evaluation"]["fixed_ppr"]]
    p_fixed = cfg["evaluation"]["fixed_ppr"][1]
    max_tpr = "; ".join(
        f"{sub} {min(1.0, p_fixed * (r['P'] + r['U']) / r['P']):.3f}"
        for sub, r in {row["subconjunto"]: row for row in size_rows}.items()
    )

    md = f"""# Informe F2.4: líneas base

> Generado por `scripts/report_f2_4.py` (commit `{git_commit()[:7]}`{", árbol sucio" if git_is_dirty() else ""}) a partir de
> `{run["run_dir"]}` (corrida con el commit `{str(run["git_commit"])[:7]}`{", árbol sucio" if bool(run["git_dirty"]) else ""}; {int(run["n_seeds"])} semillas; {run["total_s"]:.0f} s).
> No editar a mano. Configuración: `configs/experiments/{EXP}.yaml`. Solo `train_pool`.

## 1. Qué se mide

- **Punto de operación:** reglas con su umbral externo; modelos (solo E_hull, solo simetría,
  azar) con el umbral que da recall 0,90 en un bucle interno estratificado (preregistro §7.1).
- **TPR** micro-agregada sobre folds y **P(ŷ=1|U)** como control; media de {int(run["n_seeds"])} semillas e
  IC 95 % (t de Student), descriptivos. Un solo número sin intervalo significa que no varía
  entre semillas.
- Las reglas no se entrenan: cada composición recibe la misma predicción en cualquier
  esquema. Su TPR solo cambia en `temporal`, cuyo test contiene únicamente las positivas
  posteriores al corte.

Tamaño de cada subconjunto de evaluación (esquema `random`, test acumulado sobre folds):

{pd.DataFrame(size_rows).to_markdown(index=False)}

## 2. TPR en el punto de operación (líneas base principales)

{table(summary, "tpr", principal)}

## 3. P(ŷ = 1 | U) en el punto de operación (principales)

{table(summary, "ppr_u", principal)}

## 4. TPR a P(ŷ = 1) fija = {fixed[1]}

Por fold, a P(ŷ=1) exacta según la distribución de puntuaciones del test (sin etiquetas); si
el corte cae en un bloque de puntuaciones empatadas (balance de carga binario, E_hull = 0), el
bloque entra en la fracción justa (valor esperado al romper los empates al azar). TPR máxima
alcanzable si todas las predicciones positivas fueran P: {max_tpr}.

{table(summary, f"tpr_at_ppr_{fixed[1]}", principal)}

## 5. Sensibilidades

TPR:

{table(summary, "tpr", sensitivity)}

P(ŷ = 1 | U):

{table(summary, "ppr_u", sensitivity)}

## 6. Lectura (descriptiva; las pruebas formales son de F4)

- **Solo simetría (control de procedencia de H3):** en `random`, TPR {sym_r:.3f} con
  P(ŷ=1|U) = {sym_u:.3f}. El grupo espacial de la entrada de MP casi basta para separar P de U:
  las P son en su mayoría estructuras experimentales distorsionadas y las U, prototipos cúbicos
  hipotéticos. Es la procedencia de la estructura, no la sintetizabilidad. Fija la referencia
  que el modelo de grafos debe superar en H3.
- **Solo E_hull:** TPR {eh_r:.3f} con P(ŷ=1|U) = {eh_u:.3f} en `random`. Confirma la circularidad
  de la E_hull que se vio en F2.3.
- **τ:** TPR {tau_r:.3f} con P(ŷ=1|U) = {tau_u:.3f} (in-sample en la TableS1). Predice
  perovskita en muchas más U que los modelos con información de MP.
- **Caída máxima de TPR de `random` a un esquema confirmatorio**, en las líneas base
  aprendidas: E_hull {learned_drop["ehull_model"]:.3f} y simetría {learned_drop["symmetry_model"]:.3f}. Son
  modelos de una o dos entradas, que apenas dependen de la química del grupo reservado.
- **Azar:** TPR ≈ P(ŷ=1), como se espera. Es la referencia inferior de todas las curvas.
- **Cuidado al leer la sección 4:** en las reglas, la TPR a P(ŷ=1) fija cambia entre esquemas
  aunque cada composición reciba siempre la misma predicción. No es extrapolación: la P(ŷ=1)
  se fija dentro de cada fold, y la proporción de P de los folds cambia con el esquema. Por
  eso las comparaciones entre modelos a igual P(ŷ=1) se hacen pareadas, fold a fold (§7.2).
- **Puntuaciones discretas:** el modelo de solo simetría da pocas puntuaciones distintas (una
  por combinación de categorías), así que su umbral calibrado cae en bloques empatados y su
  P(ŷ=1|U) varía más entre semillas.
"""
    out = RESULTS_DIR / EXP / "informe.md"
    out.write_text(md, "utf-8")
    # Copia del resumen de la configuración usada, para leer el informe sin abrir el JSON.
    (RESULTS_DIR / EXP / "config_used.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), "utf-8"
    )
    print(md)


if __name__ == "__main__":  # requisito en Windows (regla 2.2)
    main()
