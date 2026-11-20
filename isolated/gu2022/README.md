# Entorno aislado de F3.5: modelo de grafos de Gu et al. (2022)

Código para reevaluar el modelo de Gu et al. **fuera del paquete `perovskite_pu`**. Sus pesos
(`.pth.tar`) solo se cargan aquí, con `torch.load(..., weights_only=True)`; su `data_cache.pkl`
no se usa (los grafos se recalculan desde los CIF). Nada de este entorno lee `data/processed/`
ni el conjunto reservado.

## Montaje (no versionado: se reconstruye)

```bash
mkdir -p /home/user/gu_isolated/{repo,work}
# Extraer el zip de Gu SIN .pkl ni .pyc en repo/ (ver bitácora del 2026-11-15).
cd /home/user/gu_isolated && uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python numpy scikit-learn pymatgen
```

Versiones usadas: Python 3.12, PyTorch 2.15.0+cpu, pymatgen 2026.10.24, scikit-learn 1.9.1,
numpy 2.5.4. Los hosts `download.pytorch.org` y `download-r2.pytorch.org` se añadieron al
entorno de la nube con aprobación de la autora.

`shim/torch_scatter/`: módulo de compatibilidad (solo `scatter_mean`, `scatter_min` y
`scatter`, con `torch.scatter_reduce`), porque `torch_scatter` no tiene ruedas en PyPI. Se
valida en `scripts/time_one_model.py` frente a bucles ingenuos.

## Scripts (`scripts/`; se ejecutan con `.venv/bin/python -I`)

| Script | Qué hace | Resultado (2026-11-15) |
|---|---|---|
| `time_one_model.py` | Grafos de la partición 0, fidelidad con sus pesos y coste de entrenar 1 modelo | 2,3 min por modelo (50 épocas, 4 hilos) |
| `check_fidelity.py` | Sus 94 puntuaciones de test, alineadas por cristal | Coinciden hasta 1e-6 |
| `reproduce_ensemble.py` | Ensemble de sus 100 modelos sobre los 12 907 cristales | CL score hasta 2e-7; TPR 0,957 |
| `s2_splits.py` | Variante (b) de S2: mismo test, sin polimorfos del test en train/val | 38 cristales menos por partición |
| `train_split.py` | Entrena un modelo como `2Train.py` (CPU, semillas fijas, test por cristal) | — |
| `run_s2.sh` | Orquesta S2 (100 + 100 modelos), reanudable | 100 + 100 modelos (2026-11-17) |
| `provenance_features.py` | Grupo espacial, Pm-3m y ángulo B–X–B mínimo de los 12 907 CIF (solo rasgos) | 0 errores; ~2 min con 3 procesos (2026-11-19) |
| `provenance_report.py` | Líneas base de solo simetría y de solo elementos con el protocolo de sus 100 particiones | TPR a su P(ŷ=1\|U): simetría 0,30–0,39; elementos A, B, X 0,81; su GCNN 0,957 |
| `h3_train.py` | H3 y S3: su GCNN desde cero con bagging PU (20 bolsas) en nuestros folds; lee el paquete de `scripts/h3_structures.py`; CPU o GPU (`--device`), reanudable | Pendiente de la GPU del grupo |

Las puntuaciones de test de S2 (`work/S2_{a,b}/test_xxx.json`) se copian a
`results/f3_5_gu/` y se analizan con `scripts/report_f3_5_s2.py` del proyecto.
