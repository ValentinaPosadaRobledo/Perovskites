# F3.5 — Plan de reevaluación del modelo de grafos de Gu et al. (2022) (borrador)

## Lo que dice su código (leído como texto desde el zip; nada ejecutado ni extraído)

- `1PU_Split.py`: 943 P entre 12 907 cristales. 100 particiones PU: el **mismo test**
  (94 P + 94 U) y validación (94 + 94) en todas; en cada una, 755 P + 755 U de entrenamiento
  (U muestreadas sin balancear con el resto). Barajado con `random.shuffle` sin semilla.
- `2Train.py`: GCNN propio (`gcnn/model.py`), **1 capa de convolución, 64 features**,
  batch 512, Adam lr 1e-3, **50 épocas**; aprendizaje por transferencia desde
  `base_weights/W0_xxx.pth.tar` (entrenados con datos de MP), con embedding y primera capa
  congelados; guarda el modelo con mejor AUC de validación. `cuda = True` fijo en el código.
- `gcnn/data.py`: grafos con `VoronoiNN` de pymatgen (radio 7 Å, expansión gaussiana) a
  partir de los CIF; cachea en `data_cache.pkl` (el zip trae uno: **no se usa**; se recalcula
  desde los CIF dentro del entorno aislado).
- Dependencias sin versiones (README): Python 3, PyTorch, `torch_scatter` (`scatter`,
  `scatter_mean`, `scatter_min`), pymatgen, scikit-learn, numpy.

## Coste estimado (a medir)

Cada modelo entrena con 1 510 grafos pequeños × 50 épocas: en CPU, previsiblemente
**minutos por modelo** (estimación; hay que medirlo). La construcción de los 12 907 grafos con
VoronoiNN es un coste único (estimo ~30–60 min en CPU).

| Paso | Entrenamientos | CPU (estimado) | ¿GPU? |
|---|---|---|---|
| 1. Reproducir su partición aleatoria con sus 100 modelos ya entrenados (solo inferencia) | 0 | < 1 h | No |
| 1b. Reentrenar sus 100 modelos (reproducción completa) | 100 | ~2–8 h | No |
| 2. S2: misma partición agrupando cristales por composición | 100 (o menos, declarado) | ~2–8 h | No |
| 3. H3: nuestras particiones (10 semillas × 105 folds) con bagging del grafo | 1050 × n_bolsas | con 10 bolsas, ~350 h | **Sí, o un protocolo reducido** |

Para H3 no hace falta bucle interno de hiperparámetros ni de umbral: se usan los
hiperparámetros de Gu y la comparación es a igual P(ŷ=1) que el PU `pre_dft` (§7.2), así que
el grafo solo tiene que ordenar. Aun así, `loeo_A` (69 folds por semilla) domina el coste.

## Entorno aislado (regla de seguridad: pesos y pickles solo ahí)

- Directorio fuera del repositorio (p. ej. `/home/user/gu_isolated/`), con su propio venv de
  uv; nunca se importa desde `perovskite_pu` ni ve `data/processed/` ni el holdout. Entra solo
  el zip de Gu (público) y, para el paso 3, una tabla de composiciones y folds de `train_pool`
  (sin el holdout).
- PyTorch para CPU: el índice oficial de CPU (`download.pytorch.org`) no está en la lista de
  hosts permitidos; el de PyPI trae dependencias CUDA de varios GB. Habría que añadir ese host
  (con tu aprobación) o aceptar la descarga grande.
- `torch_scatter`: en PyPI solo hay código fuente (compilarlo tarda) y las ruedas están en
  `data.pyg.org` (otro host). Alternativa: un módulo de compatibilidad con
  `torch.scatter_reduce` (sum, mean, amin), validado con tests; se declara como cambio mínimo.
- `torch.load` con `weights_only=True` (PyTorch ≥ 2.6) para sus pesos: no ejecuta código
  arbitrario al cargar un `state_dict`.
- Cambios mínimos a su código, documentados en un parche: CPU en vez de `cuda`, semillas
  fijas, y (si hace falta) el módulo de compatibilidad de `torch_scatter`.

## Decisiones para la autora

- **G1.** Aprobar el paso de medición: instalar PyTorch (CPU) y dependencias en el entorno
  aislado, construir los grafos y cronometrar 1 modelo. Implica elegir entre añadir el host
  de PyTorch para CPU o la descarga grande de PyPI.
- **G2.** ¿Hay GPU del grupo (modelo y memoria)? Decide si el paso 3 se hace con el protocolo
  completo o reducido.
- **G3.** Protocolo reducido para H3 si no hay GPU (a decidir ANTES de ver resultados del
  grafo; se registra en la bitácora y se declara como desviación): menos bolsas (p. ej. 5–10)
  manteniendo las 10 semillas y los 4 esquemas confirmatorios.
- **G4.** Estructura por composición en nuestras particiones (paso 3): criterio explícito
  (p. ej. la entrada experimental de menor E_hull para P, la de menor E_hull para U). Es
  donde entra el control de procedencia (línea base de solo simetría).
