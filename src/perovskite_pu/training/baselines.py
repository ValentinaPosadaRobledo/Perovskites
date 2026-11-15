"""F2.4: líneas base de reglas y aprendidas.

Entradas por composición (`baseline_inputs`):
- `tau`, `t`: de las features `pre_dft_tau` (algoritmo de Bartel, solo la fórmula);
- `ehull`: E_hull GGA del polimorfo perovskita (features `post_dft`);
- `spacegroup_number`, `crystal_system`: de la MISMA entrada de MP que `post_dft`. Solo los usa
  la línea base de simetría, que es el control de procedencia de la estructura (H3). No forman
  parte de ninguna familia de features (ver `configs/feature_audit.yaml`);
- `charge_balanced_icsd`: balance de carga con los estados observados en ICSD (bandera de F1.5);
- `charge_balanced_common`: con los estados comunes de pymatgen (`compound possible`);
- subconjuntos de evaluación: `tau_defined`, `t_defined`, `post_dft` y `all`.

Reglas (sin entrenamiento): la puntuación ordena de forma coherente con la regla y el umbral
es externo. Modelos (con entrenamiento): estimadores de sklearn cuyo umbral se calibra en el
bucle interno, como en los modelos PU.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Subconjuntos de evaluación admitidos en el YAML (columnas booleanas de `baseline_inputs`).
SUBSETS = ("all", "tau_defined", "t_defined", "post_dft")


def baseline_inputs(compositions: Sequence[str], pool: pd.DataFrame | None = None) -> pd.DataFrame:
    """Tabla de entradas de las líneas base, en el orden de `compositions` (train_pool).

    `pool` (con columna `composition`) es el pool de la corrida, de donde se lee la bandera de
    balance de carga; por defecto, `train_pool`. Las sensibilidades con un pool aumentado
    (F3.6, sensibilidad 13) pasan el suyo.
    """
    from perovskite_pu.features import sets
    from perovskite_pu.utils.paths import INTERIM_DIR, PROCESSED_DIR, ROOT, TRAIN_POOL_FILENAME

    comps = list(compositions)
    paths = sets.output_paths()  # rutas de features.output (configs/data.yaml)
    if pool is None:
        pool = pd.read_parquet(PROCESSED_DIR / TRAIN_POOL_FILENAME)
    pool = pool.set_index("composition")
    pre = sets.load_feature_matrix("pre_dft_tau", comps)
    pre_meta = pd.read_parquet(ROOT / paths["metadata"])
    post_meta = pd.read_parquet(ROOT / paths["post_metadata"])
    entries = pd.read_parquet(INTERIM_DIR / "f1_5_entries.parquet").set_index("material_id")

    out = pd.DataFrame(index=pd.Index(comps, name="composition"))
    # Factores geométricos (solo la fórmula) y balance de carga con dos definiciones.
    out["tau"] = pre["geom_tau"]
    out["t"] = pre["geom_t"]
    no_balance = pool.loc[comps, "flag_no_charge_balance"].astype(bool)
    out["charge_balanced_icsd"] = (~no_balance).astype(float).to_numpy()
    out["charge_balanced_common"] = pre["compound possible"].astype(float)

    # Termodinámica y simetría: solo en las composiciones elegibles para post_dft.
    ok = sets.eligible("post_dft", comps)
    out["post_dft"] = ok.to_numpy()
    post = sets.load_feature_matrix("post_dft", [c for c in comps if ok[c]])
    out["ehull"] = post["thermo_ehull"].reindex(comps)
    mid = post_meta["material_id_gga"].reindex(comps).where(out["post_dft"])
    sg = mid.map(entries["spacegroup_number"])
    out["spacegroup_number"] = sg.map(lambda v: f"{int(v)}" if pd.notna(v) else np.nan)
    out["crystal_system"] = mid.map(entries["crystal_system"])

    # Subconjuntos de evaluación.
    out["all"] = True
    out["tau_defined"] = out["tau"].notna()
    out["t_defined"] = out["t"].notna()
    # Pertenencia a la TableS1 de Bartel: para la sensibilidad de H2 sin τ in-sample.
    in_s1 = pre_meta["in_bartel2019"].reindex(comps)
    out["in_bartel2019"] = in_s1.astype("boolean").fillna(False).astype(bool).to_numpy()
    return out


# ---------------------------------------------------------------------------
# Reglas
# ---------------------------------------------------------------------------
def rule_scores(x: np.ndarray, rule: Mapping[str, Any]) -> tuple[np.ndarray, float]:
    """Puntuación y umbral (en unidades de la puntuación) de una regla.

    - `below`: positivo si x ≤ umbral → puntuación −x, umbral −umbral.
    - `above`: positivo si x ≥ umbral → puntuación x.
    - `window`: positivo si a ≤ x ≤ b → puntuación −|x − (a+b)/2|, umbral −(b−a)/2. Así la
      puntuación también ordena fuera de la ventana (para las curvas TPR frente a P(ŷ=1)).
    """
    x = np.asarray(x, dtype=float)
    kind = rule["type"]
    if kind == "below":
        return -x, -float(rule["threshold"])
    if kind == "above":
        return x, float(rule["threshold"])
    if kind == "window":
        a, b = (float(v) for v in rule["window"])
        if not a < b:
            raise ValueError(f"Ventana inválida: {rule['window']}")
        return -np.abs(x - (a + b) / 2), -(b - a) / 2
    raise ValueError(f"Tipo de regla desconocido: {kind}")


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
class RandomScores(ClassifierMixin, BaseEstimator):
    """Clasificador de azar: puntuaciones uniformes independientes de las features.

    Es la referencia trivial: con cualquier umbral, la TPR esperada es igual a P(ŷ=1).
    Las puntuaciones son reproducibles con `random_state`.
    """

    def __init__(self, random_state: int | None = None):
        self.random_state = random_state

    def fit(self, X, y):
        # Solo fija el generador; no aprende nada de X ni de y.
        self.classes_ = np.array([0, 1])
        self._rng = np.random.default_rng(self.random_state)
        return self

    def predict_proba(self, X):
        p = self._rng.random(len(X))
        return np.column_stack([1 - p, p])


def make_estimator(
    estimator: str, inputs: Sequence[str], params: Mapping[str, Any], random_state: int
) -> BaseEstimator:
    """Estimador de sklearn de una línea base aprendida (todo dentro de un Pipeline)."""
    if estimator == "logistic":
        return Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=params.get("C", 1.0), max_iter=1000,
                                       random_state=random_state)),
        ])  # fmt: skip
    if estimator == "logistic_onehot":
        enc = OneHotEncoder(handle_unknown="infrequent_if_exist",
                            min_frequency=params.get("min_frequency", 5))  # fmt: skip
        return Pipeline([
            ("onehot", ColumnTransformer([("cat", enc, list(range(len(inputs))))])),
            ("clf", LogisticRegression(C=params.get("C", 1.0), max_iter=1000,
                                       random_state=random_state)),
        ])  # fmt: skip
    if estimator == "random_scores":
        return RandomScores(random_state=random_state)
    raise ValueError(f"Estimador desconocido: {estimator}")


def model_matrix(inputs_table: pd.DataFrame, inputs: Sequence[str]) -> np.ndarray:
    """Matriz de entrada de un modelo (una columna ficticia si no tiene entradas)."""
    if not inputs:
        return np.zeros((len(inputs_table), 1))
    return inputs_table[list(inputs)].to_numpy()
