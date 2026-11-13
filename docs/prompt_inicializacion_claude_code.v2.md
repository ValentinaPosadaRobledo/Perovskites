Vas a inicializar un repositorio de investigación vacío. Este proyecto es una tesis de maestría que además debe producir un artículo científico. Lee todo este mensaje antes de actuar. En esta primera sesión **solo inicializas** el repositorio: no descargues datos ni entrenes modelos.

> **Versión 2 (2026-11-12).** Este prompt incorpora la revisión de cinco especialistas (química de estado sólido, química computacional, machine learning, MLOps y metodología de la investigación). La versión original está en `prompt_inicializacion_claude_code.v1.md`. Los cambios principales están resumidos en la sección 7.

---

# 1. Contexto del proyecto

**Título de trabajo:** Evaluación de la generalización de modelos de aprendizaje positivo–no etiquetado (PU learning) para predecir la sintetizabilidad de perovskitas óxido ABO₃ con control de fuga de información.

**Problema.** Las bases de datos de materiales (Materials Project, OQMD, AFLOW) contienen miles de perovskitas hipotéticas, pero solo una fracción se ha sintetizado. Los modelos de sintetizabilidad publicados se han evaluado sobre todo con particiones aleatorias, que sobreestiman el rendimiento cuando se quiere predecir química nueva (elementos o familias químicas no vistas).

**Brecha (delimitada).** Las particiones fuera de distribución ya existen como herramientas genéricas: LOCO-CV (Meredig et al., 2018) y MatFold (Witman y Schindler, *Digital Discovery*, 2025). También hay trabajos de PU learning para sintetizabilidad que abordan la generalización: SynCoTrain (arXiv 2411.12011) y Desai et al. (arXiv 2510.06166). La contribución de este proyecto **no** es un algoritmo nuevo. Es una **evaluación sistemática y preregistrada de la sintetizabilidad PU en perovskitas óxido bajo extrapolación química**, con reevaluación del modelo de grafos de Gu et al. (2022). *Todas estas referencias deben verificarse y leerse antes de redactar la introducción (tarea F0.5).*

**Qué se predice (definición operativa).** La tarea es **sintetizabilidad como perovskita**: dada una composición ABO₃, ¿existe un polimorfo con estructura perovskita (octaedros BO₆ que comparten solo vértices) reportado experimentalmente?

- **Positivo:** la composición tiene al menos un polimorfo **perovskita** con registro experimental en ICSD (`theoretical == False` en Materials Project, contrastado con la presencia de `database_IDs["icsd"]`).
- **Sintetizada no perovskita:** la composición está en ICSD, pero **solo** en estructuras no perovskita (ilmenita, LiNbO₃, YMnO₃ hexagonal, etc.). **No** es positivo ni no etiquetado: se guarda aparte y se usa en la validación de formabilidad (sección 2.3).
- **No etiquetado:** sin registro experimental. **Los no etiquetados NO son negativos**: el problema es PU, no binario.
- **Banderas de sensibilidad:** perovskitas de alta presión o metaestables (p. ej., MgSiO₃, BiCoO₃, PbVO₃, BiScO₃) y casos con estados de oxidación ambiguos (BiNiO₃, CaFeO₃). Se conservan en el análisis principal, se marcan y se repite el análisis sin ellas.

**Alcance y delimitaciones.**

- Solo óxidos ABO₃ ordenados y estequiométricos presentes en Materials Project.
- Quedan fuera las soluciones sólidas, las fases dopadas, las de ocupación parcial y las deficientes en oxígeno (MP no las contiene).
- No hay validación experimental propia.
- Las estructuras de ICSD **no se redistribuyen**, por su licencia; solo se publican identificadores y features derivadas.

**Referencia principal que se reevalúa:** Gu et al. (2022), *npj Computational Materials* 8:71, "Perovskite synthesizability using graph neural networks". Reportaron 943 perovskitas sintetizadas y 11.964 virtuales (todas las clases), con TPR = 0,957 evaluado sobre un 10% aleatorio de positivos. Datos y código: https://doi.org/10.5281/zenodo.6348980

## 1.1 Objetivos

**Objetivo general.** Evaluar la capacidad de generalización de modelos de aprendizaje positivo–no etiquetado para predecir la sintetizabilidad de perovskitas óxido ABO₃ bajo esquemas de partición con control de fuga, y explorar su uso para priorizar candidatos estables en medio acuoso.

**Objetivos específicos.**

- **OE1.** Construir un conjunto de datos de perovskitas ABO₃ con estas características:
  - identificadas por conectividad estructural;
  - deduplicadas;
  - etiquetadas a nivel de composición;
  - con particiones aleatoria, por elemento (A y B), por familia química, por clúster químico y, si los datos lo permiten, temporal;
  - con un conjunto de prueba externo congelado y definido por grupos.
- **OE2.** Cuantificar la caída de rendimiento entre la partición aleatoria y las particiones por grupos. Se mide para modelos PU composicionales, para el modelo de grafos de Gu et al. y para las líneas base (t, τ, E_hull, solo grupo espacial). Alimenta H1 y H3.
- **OE3.** Determinar si un modelo PU composicional e interpretable es no inferior al factor de tolerancia τ bajo partición por grupos, e identificar sus variables determinantes y la estabilidad de estas entre semillas (SHAP). Alimenta H2.
- **OE4 (exploratorio, solo tesis).** Priorizar candidatos con alta sintetizabilidad predicha según su estabilidad termodinámica acuosa (ΔG de Pourbaix) en condiciones fisiológicas, con filtro de toxicidad y discusión explícita de sus limitaciones.

## 1.2 Hipótesis (operacionalizadas)

Los márgenes δ y los tests se **preregistran** antes de abrir el holdout (tarea F0.4). Los valores indicados son propuestas iniciales que el autor debe confirmar en el preregistro.

- **H1 (efecto de la partición).** El TPR, medido en un punto de operación fijo, cae al pasar de `random` a cada esquema por grupos (`loeo_A`, `loeo_B`, `family`, `cluster`). El punto de operación es el umbral elegido en el bucle interno, y además se reporta el TPR a P(ŷ = 1) fija.
  - *Criterio:* la caída es relevante si el IC 95% de ΔTPR (bootstrap jerárquico por grupo) excluye 0 **y** la estimación puntual es ≥ δ₁ = 0,05.
  - *Resultado nulo:* es publicable. Indicaría que la sobreestimación es menor de lo esperado.
- **H2 (no inferioridad frente a τ).** Bajo los esquemas por grupos, el mejor modelo PU `pre_dft` es **no inferior** a τ de Bartel et al. (2019) en TPR a igual P(ŷ = 1).
  - *Criterio:* test unilateral de no inferioridad con margen δ₂ = 0,03, es decir, la cota inferior del IC 95% de (TPR_PU − TPR_τ) es > −δ₂.
  - La superioridad se reporta solo si la cota inferior es > 0.
  - Como τ se ajustó sobre los datos de Bartel, toda evaluación de τ sobre ese subconjunto se declara **in-sample**.
- **H3 (grafos frente a composición bajo extrapolación).** La ventaja del modelo de grafos sobre el mejor modelo composicional se reduce al exigir extrapolación.
  - *Criterio:* [ΔTPR(grafo − composicional)]\_random − [ΔTPR(grafo − composicional)]\_grupos > 0, con un IC 95% que excluya 0.
  - *Control obligatorio:* una línea base que use **solo el grupo espacial o la simetría**. Las estructuras experimentales suelen estar distorsionadas y las hipotéticas suelen ser prototipos cúbicos, así que un modelo de grafos podría aprender la procedencia de la estructura en lugar de la sintetizabilidad.
  - Si la reproducción del modelo de grafos falla dentro del plazo fijado en F3.5, H3 se reformula como pregunta exploratoria y se registra en la bitácora.

**OE4** se plantea como **objetivo exploratorio**, no como hipótesis: un enunciado de existencia ("es posible identificar…") no es falsable.

## 1.3 Productos esperados

- **Artículo:** OE1–OE3 (H1–H3). El núcleo es un benchmark de sintetizabilidad con control de fuga. **Revista objetivo:** *Digital Discovery* (RSC); alternativa: *JCIM* (ACS).
- **Tesis:** OE1–OE4. Incluye el artículo, el capítulo exploratorio de estabilidad acuosa y la discusión de aplicaciones biomédicas con sus limitaciones.
- **Núcleo mínimo publicable:** Fases 1–4 con H1, H2 y la reevaluación de Gu et al. bajo partición por grupos. SHAP con estabilidad entre semillas, la partición temporal y la Fase 5 son ampliaciones.

---

# 2. Reglas permanentes del proyecto

Crea un archivo `CLAUDE.md` en la raíz con estas reglas, para que se apliquen en todas las sesiones futuras.

## 2.1 Control de data leakage (obligatorio)

1. **Identificar perovskitas por conectividad**, no solo por similitud a un prototipo:
   - estados de oxidación con `BVAnalyzer` u `oxi_state_guesses`;
   - `CrystalNN`/ChemEnv: B con coordinación octaédrica (NC = 6) y A con NC entre 8 y 12;
   - octaedros BO₆ que comparten **solo vértices**, con ángulos B–O–B > ~140°;
   - contraste con `StructureMatcher` (tolerancias documentadas en `configs/data.yaml`).
   - Excluir los polianiones (carbonatos, nitratos, boratos).
   - Clasificar aparte los politipos hexagonales 2H/4H/6H (octaedros que comparten caras).
   - Validar a mano al menos 50 casos límite y documentarlos.
2. **Deduplicar antes de particionar.** Duplicados estructurales con `StructureMatcher` de pymatgen; luego agrupar por composición reducida. Las composiciones dopadas o casi idénticas se agrupan con su compuesto padre.
3. **Etiqueta a nivel de composición** (definición de la sección 1). Nunca puede aparecer la misma composición en entrenamiento y prueba.
4. **Esquemas de partición.** Siempre se reportan por separado, junto con la distribución de tamaños de grupo y de positivos por grupo:
   - `random`: partición aleatoria de composiciones (solo como referencia, para medir la fuga).
   - `group_system`: GroupKFold por sistema químico A–B. **Solo es diagnóstico.** En ABO₃ cada sistema suele contener una sola composición, así que se espera que se parezca a `random`. Hay que demostrarlo con la distribución de tamaños de grupo.
   - `loeo_A`: leave-one-A-element-out.
   - `loeo_B`: leave-one-B-element-out. Los elementos con menos de 5 positivos se fusionan por familia química o se reportan aparte.
   - `family`: leave-one-family-out del elemento B (3d, 4d/5d, 4f/5f, grupos principales). La definición de familias va en `configs/data.yaml`.
   - `cluster`: LOCO-CV por clustering (k-means sobre features MAGPIE, Meredig et al., 2018).
   - `temporal` (si hay año de ICSD disponible): entrenar con entradas anteriores a un año de corte y probar con las posteriores.
   - Usar **MatFold** para generar las particiones si es compatible. Si no lo es, implementarlas en `src/<paquete>/splits/` y documentar el motivo en la bitácora.
5. **Conjunto de prueba externo congelado, definido por grupos.**
   - Contenido: todas las composiciones cuyo elemento B pertenece a un conjunto reservado de elementos, elegidos de forma estratificada por familia hasta cubrir ~10–15% de los positivos.
   - Se crea una sola vez con `holdout_seed: 2026`. Es independiente de las semillas de los experimentos.
   - Se guarda en `data/splits/holdout_frozen.csv` en forma **canónica** (composiciones ordenadas, fin de línea LF, UTF-8). Su SHA-256 se registra en `data/splits/HASHES.json` (legible por código) y se publica en el preregistro.
   - En el mismo paso se genera `data/processed/train_pool.parquet`, que **excluye** el holdout. Todo el entrenamiento y el ajuste de hiperparámetros parten de `train_pool`.
   - El holdout solo se lee con una única función `load_holdout()`, que exige la variable de entorno `HOLDOUT_UNLOCK=1`. Ningún módulo de `src/<paquete>/training/` puede importarla.
   - Si hay datos temporales, se añade un holdout temporal complementario.
6. **Todo el preprocesamiento dentro de un `sklearn.pipeline.Pipeline`** (imputación, escalado, selección de features), ajustado solo con datos de entrenamiento de cada fold. Si τ se usa como feature, los compuestos del conjunto de Bartel se excluyen de la evaluación o τ se reajusta dentro de cada fold.
7. **Validación cruzada anidada.**
   - Bucle interno: hiperparámetros, con búsqueda aleatoria de presupuesto fijo, ~10–20 configuraciones definidas en el YAML del experimento. Aquí también se elige el **umbral de decisión**.
   - Bucle externo: para reportar.
8. **Auditoría de features.** Ninguna feature puede derivarse de la etiqueta ni de información posterior a la síntesis. Hay dos familias de modelos que se reportan por separado:
   - `pre_dft`: solo composición y factores geométricos (MAGPIE, t, μ, τ).
   - `post_dft`: lo anterior más `energy_above_hull` y `formation_energy_per_atom`.
     - **Advertencia de circularidad:** el casco convexo de MP lo definen en parte las propias fases de ICSD, y las estructuras hipotéticas suelen ser prototipos sin relajar sus inclinaciones octaédricas. Por eso E_hull está correlacionada con la etiqueta casi por construcción.
     - `post_dft` se reporta como **cota superior informada por el hull**, junto con una línea base que use solo E_hull.
     - Usar una única funcional (`thermo_types=["GGA_GGA+U"]`) y analizar la sensibilidad con r2SCAN.
9. **Tests automáticos de fuga** en `tests/test_leakage.py`, que deben pasar antes de cualquier entrenamiento. Están organizados en capas:
   - **Siempre activos (con fixtures sintéticas en `tests/fixtures/`):**
     - no hay solapamiento de composiciones entre train y test en ningún fold de ningún esquema;
     - no hay solapamiento de grupos en los esquemas por grupos;
     - el preprocesamiento se ajusta solo con train;
     - análisis estático (`ast`) que verifica que ningún archivo de `src/<paquete>/training/` referencia `holdout` ni importa `load_holdout`;
     - durante una corrida con datos sintéticos, un `sys.addaudithook` sobre el evento `open` falla si se abre `holdout_frozen.csv`.
   - **Dependientes de datos** (`skipif` sin datos): intersección vacía entre las composiciones del holdout y las de `train_pool`; el hash canónico del holdout coincide con `HASHES.json`.
     - Si `REQUIRE_DATA=1` o existe `data/processed/`, estos tests **fallan** en lugar de saltarse.
     - Ejecutar siempre con `pytest -rs`.
     - Ninguna tarea a partir de F1.7 se cierra con tests de fuga saltados.

## 2.2 Múltiples semillas y estadística (obligatorio)

- Las semillas se definen en un solo lugar: `configs/seeds.yaml`, con `seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]` y `holdout_seed: 2026`.
- **Cada experimento se ejecuta con todas las semillas.**
  - De cada semilla se derivan flujos independientes con `np.random.SeedSequence(seed).spawn(3)`: asignación de folds, muestreo de no etiquetados en el bagging PU e inicialización de los modelos.
  - En los esquemas leave-one-group-out los folds son deterministas: ahí la semilla solo afecta al bagging y al modelo, y así se declara.
- **Desarrollo incremental:** primero 3 semillas para medir tiempos. Las 10 semillas son obligatorias antes de marcar una tarea como completada.
- **Resultados:**
  - Cada corrida en `results/<experimento>/<timestamp>_<hash_config>/`, con `raw/seed_<n>.csv` (por semilla y fold) y `run_metadata.json`.
  - Agregado en `results/<experimento>/summary.csv`, con media, desviación estándar e IC 95%.
  - Índice de corridas en `results/runs_index.csv`.
- **Tests estadísticos:**
  - CV repetida con folds aleatorios (`random`, `cluster`): t-test corregido para remuestreo de Nadeau y Bengio. No usar el t-test pareado simple.
  - Esquemas leave-one-group-out (`loeo_A`, `loeo_B`, `family`): **bootstrap jerárquico por grupo**, o tests de permutación por grupo. Nadeau–Bengio no aplica.
  - H2: **test de no inferioridad** con margen preregistrado.
  - Comparaciones múltiples: corrección de Holm.
- **`run_metadata.json`** registra:
  - commit de git; la corrida se **niega a ejecutarse con el árbol sucio**, salvo con `--allow-dirty`, en cuyo caso guarda el diff;
  - fecha;
  - configuración completa y su hash;
  - versiones de paquetes y hash de `uv.lock`;
  - semillas;
  - versión de la base de datos de MP;
  - CPU, `n_jobs` y tiempos de ejecución.
- **Paralelización:** joblib (`loky`) solo en el nivel externo, semilla × fold. `n_jobs=1` en RF/GB y `OMP_NUM_THREADS=1`. En Windows, todo punto de entrada usa `if __name__ == "__main__":`.

## 2.3 Métricas para PU learning

- **Principal:** recall (TPR) sobre positivos reservados, en un punto de operación fijo.
- **Control:** fracción de no etiquetados predichos como positivos, P(ŷ = 1 | U).
- **Curvas:** TPR frente a P(ŷ = 1) para todos los modelos y líneas base. También se reporta el TPR a varias P(ŷ = 1) fijas, para comparar modelos con umbrales distintos (incluido τ, cuyo umbral de 4,18 viene de fuera).
- **Comparación entre modelos:** métrica de Lee y Liu (2003), recall² / P(ŷ = 1). Supone SCAR (selección completamente aleatoria de los positivos etiquetados), que aquí **no se cumple**: hay sesgo químico en lo que se ha sintetizado. Por eso solo sirve para **ordenar** modelos, no como estimación absoluta.
- **Prior de clase:** estimar π con al menos dos métodos (Elkan–Noto, TIcE o KM2) y hacer análisis de sensibilidad.
- **Validación de formabilidad (no de sintetizabilidad):**
  - Se usan los negativos experimentales de Bartel et al. (2019) y las composiciones "sintetizadas no perovskita" de la sección 1.
  - Estos datos miden si un compuesto **adopta la estructura perovskita**, no si es sintetizable. Se reportan con ese nombre.
  - Antes de usarlos, comprobar que no se solapan con los positivos.
  - Los resultados de τ sobre los datos de Bartel se marcan como *in-sample*.
- **Cotas inferiores:** accuracy, F1 y ROC-AUC que traten los no etiquetados como negativos solo pueden reportarse como cotas inferiores y etiquetarse así.
- **`loeo_B` y `loeo_A`:** TPR micro-agregado sobre todos los folds, más la distribución por elemento.

## 2.4 Reproducibilidad, entorno y estilo

- **Entorno:** Python 3.12 (`requires-python = ">=3.12,<3.13"`), gestionado con **uv**.
  - Se versionan `uv.lock` y `.python-version`. Todo se ejecuta con `uv run`.
  - **Respaldo:** Poetry solo si uv no logra resolver o instalar las dependencias en el sistema. Ejemplos: conflictos de `matminer` o `emmet-core` con numpy/pandas, o falta de wheels en Windows. En ese caso, registrar el error exacto y la decisión en la bitácora antes de cambiar.
- **Paquete:** layout `src/` con nombre propio, `src/perovskite_pu/`. Los subpaquetes no se llaman `src.data`.
- **Tareas:** `poethepoet` (`[tool.poe.tasks]` en `pyproject.toml`) para `test`, `lint`, `figures`, etc. En Windows no hay `make`.
- **Calidad:**
  - `ruff` para lint y formato.
  - `pre-commit` con `ruff`, `detect-secrets`, `nbstripout` y `check-added-large-files`.
  - CI en GitHub Actions (`astral-sh/setup-uv`) que ejecuta `ruff` y `pytest -rs` con fixtures sintéticas.
- **Secretos:**
  - La clave de Materials Project se lee de la variable de entorno `MP_API_KEY` desde un archivo `.env` que **nunca** se versiona.
  - Crear `.env.example` sin la clave.
- **Datos:**
  - Los datos crudos van en `data/raw/` (caché en parquet o `json.gz`) y no se versionan. Se versiona el script que los descarga.
  - `data/MANIFEST.json` registra el SHA-256 de cada archivo crudo, la versión de la base de datos de MP (`MPRester().get_database_version()`), la versión de `mp-api` y `emmet-core` y la fecha de la consulta.
  - `.gitattributes` con `*.csv -text` y `*.parquet binary`, para que los hashes no cambien por la conversión CRLF/LF en Windows.
- **Configuración:** todo experimento se lanza desde un archivo de configuración en `configs/`, no con parámetros escritos en el código.
- **Figuras:**
  - Cada figura y tabla del artículo se genera con un script en `paper/scripts/`, a partir de `results/`. Nada se edita a mano.
  - Cada figura va acompañada de un `.json` con los hashes de los `summary.csv` de entrada y el commit.
  - `poe figures` regenera todas.
- **Idioma:**
  - Código, nombres de variables, funciones y comentarios en inglés.
  - Documentos del proyecto (ROADMAP, outlines, bitácora) y mensajes de commit en español.
- **Git:**
  - Commits pequeños con mensajes descriptivos en español.
  - **Antes de cada commit, mostrar el diff y pedir confirmación explícita al usuario.**
  - No añadir trailers de coautoría de IA.
- **Publicación:**
  - Al cerrar, archivar en Zenodo el código y los datos derivados redistribuibles, con DOI.
  - Declarar el uso de asistentes de IA (Claude Code) según la política de la revista. La IA no figura como autora.

## 2.5 Mantenimiento del ROADMAP

- Antes de empezar una tarea, cambia su estado a **🔄 En proceso** y registra la fecha.
- Al terminarla, cámbiala a **✅ Completada**, con fecha, el commit **que produjo el resultado** y el enlace al resultado o archivo producido. La actualización del ROADMAP va en un commit aparte (un commit no puede contener su propio hash).
- Si una tarea queda bloqueada, mantenla en 🔄 y escribe el motivo en la columna de notas.
- Nunca marques como completada una tarea cuyos tests fallen o se salten, o cuyos resultados no estén guardados para todas las semillas.
- Registra cada decisión metodológica relevante en la sección "Bitácora de decisiones" del ROADMAP, con fecha y justificación.
- Opcional: `scripts/roadmap.py done <ID>` rellena la fecha y `git rev-parse --short HEAD`.

## 2.6 Preregistro

- `docs/preregistro.md` contiene:
  - definición de positivo;
  - esquemas de partición;
  - definición del holdout y su hash;
  - hipótesis con sus márgenes δ₁ y δ₂;
  - tests estadísticos;
  - punto de operación;
  - análisis de sensibilidad previstos.
- Se congela y se deposita en OSF **antes** de la fase F4 (evaluación sobre el holdout).
- Cualquier desviación posterior se declara explícitamente en el artículo.

---

# 3. Estructura del repositorio

```
.
├── CLAUDE.md
├── ROADMAP.md
├── README.md
├── pyproject.toml
├── uv.lock
├── .python-version
├── .env.example
├── .gitignore
├── .gitattributes
├── .pre-commit-config.yaml
├── .github/workflows/ci.yml
├── configs/
│   ├── seeds.yaml
│   ├── data.yaml             # tolerancias, familias químicas, criterios de identificación
│   └── experiments/          # un YAML por experimento (+ _template.yaml)
├── data/
│   ├── MANIFEST.json
│   ├── raw/                  # no versionado
│   ├── interim/
│   ├── processed/            # train_pool.parquet
│   └── splits/
│       └── HASHES.json
├── docs/
│   ├── preregistro.md
│   └── validacion_manual.md  # casos límite de identificación de perovskitas
├── src/perovskite_pu/
│   ├── data/                 # descarga, identificación, deduplicación, etiquetado
│   ├── features/             # MAGPIE, t, μ, τ, estados de oxidación
│   ├── splits/               # random, group_system, loeo_A, loeo_B, family, cluster, temporal, holdout
│   ├── training/             # bagging PU, modelos, CV anidada (NO puede leer el holdout)
│   ├── evaluation/           # métricas PU, prior de clase, agregación, tests estadísticos, load_holdout()
│   ├── pourbaix/             # extensión de tesis (OE4)
│   └── utils/                # semillas, metadatos de corrida, hashing canónico
├── scripts/
│   └── roadmap.py            # opcional
├── tests/
│   ├── fixtures/             # datos sintéticos
│   ├── test_leakage.py
│   └── test_metrics.py
├── notebooks/                # solo exploración, nunca resultados finales
├── results/
├── paper/
│   ├── scripts/
│   ├── figures/
│   ├── tables/
│   └── manuscript_outline.md
└── thesis/
    └── chapter_outline.md
```

---

# 4. Contenido inicial de ROADMAP.md

Crea `ROADMAP.md` con esta estructura. Todas las tareas empiezan en ⬜ Pendiente, excepto F0.1–F0.3, que pasan a ✅ Completada al terminar esta sesión.

**Leyenda de estados:** ⬜ Pendiente · 🔄 En proceso · ✅ Completada

Cada fase es una tabla con columnas: `ID | Tarea | Destino | Estado | Inicio | Fin | Commit | Resultado / Notas`. La columna Destino indica si la tarea alimenta el artículo (A), la tesis (T) o ambos (A+T).

Incluye al inicio el **cronograma de referencia (~20 meses)**:

| Meses | Fases |
|---|---|
| 0–2 | F0 + revisión de literatura + preregistro (borrador) |
| 2–5 | F1, datos (mayor riesgo: suele llevar el doble de lo previsto) |
| 5–7 | F2, features y líneas base |
| 7–11 | F3, modelos PU y modelo de grafos |
| 11–13 | F4, evaluación final e interpretabilidad |
| 13–16 | Redacción y envío del artículo |
| 15–18 | F5, extensión de tesis |
| 18–20 | Redacción de la tesis y sustentación |

**Fase 0 — Inicialización (A+T)**
- F0.1 Estructura del repositorio, CLAUDE.md, ROADMAP.md, README.md, entorno uv (o Poetry como respaldo), `.gitattributes`, pre-commit y CI.
- F0.2 Esqueleto de `tests/test_leakage.py` y `tests/test_metrics.py`, con fixtures sintéticas activas y tests dependientes de datos en `skipif`.
- F0.3 Configuración de semillas (`SeedSequence`) y plantilla de configuración de experimentos.
- F0.4 Borrador de `docs/preregistro.md` (hipótesis, δ₁, δ₂, tests, punto de operación).
- F0.5 Revisión de literatura: verificar y leer MatFold, LOCO-CV, SynCoTrain, Desai et al. (2025), Gu et al. (2022), Bartel et al. (2019), Jang et al. (2020), Frey et al. (2020) y SynthNN. Delimitar la brecha.

**Fase 1 — Datos (A+T)**
- F1.1 Descarga desde Materials Project con `mp-api`:
  - campos `material_id`, `formula_pretty`, `structure`, `theoretical`, `database_IDs`, `energy_above_hull`, `formation_energy_per_atom`, `symmetry`;
  - termodinámica con `thermo_types=["GGA_GGA+U"]`;
  - registro de la versión de la base de datos en `data/MANIFEST.json`.
- F1.2 Descarga del dataset de Gu et al. (Zenodo) y del dataset de Bartel et al. (2019), con verificación de su solapamiento con MP.
- F1.3 Identificación de perovskitas ABO₃ por conectividad (CrystalNN/ChemEnv + StructureMatcher), según la regla 2.1.1:
  - exclusión de polianiones;
  - clasificación aparte de los politipos hexagonales;
  - asignación de estados de oxidación y marcado de los casos ambiguos.
- F1.4 Validación manual de ≥ 50 casos límite en `docs/validacion_manual.md`.
- F1.5 Deduplicación estructural y por composición, agrupación de casi duplicados y etiquetado a nivel de composición (positivo / sintetizada no perovskita / no etiquetado), con banderas de alta presión y metaestabilidad.
- F1.6 Informe de conteos: positivos, sintetizadas no perovskita y no etiquetados, por estructura, por composición, por elemento A y B y por familia.
  - **Punto de decisión:** si hay menos de ~100 composiciones ABO₃ positivas (estimación previa: 200–350), ampliar a todas las clases de perovskita y evaluar los óxidos por separado.
  - Registrar la decisión en la bitácora.
- F1.7 Generación de todas las particiones (MatFold o implementación propia), con informe de tamaños de grupo y de positivos por grupo. Demostrar empíricamente cuánto se parece `group_system` a `random`.
- F1.8 Creación del holdout congelado por grupos (y del temporal, si aplica), de `train_pool.parquet` y del hash canónico en `HASHES.json`. Todos los tests de fuga pasan sin saltos.

**Fase 2 — Features y líneas base (A+T)**
- F2.1 Features `pre_dft`: MAGPIE (ElementProperty, Stoichiometry, ValenceOrbital, IonProperty) + t, μ, τ (radios de Shannon, A en coordinación XII y B en VI; se documentan los radios extrapolados).
- F2.2 Features `post_dft`: lo anterior + E_hull y ΔHf (una sola funcional).
- F2.3 Auditoría de features documentada, incluido el análisis de la circularidad de E_hull.
- F2.4 Líneas base, evaluadas con todos los esquemas de partición y todas las semillas:
  - t de Goldschmidt;
  - τ de Bartel;
  - umbral de E_hull;
  - modelo que usa solo E_hull;
  - modelo que usa solo el grupo espacial o la simetría;
  - clasificador trivial.

**Fase 3 — Modelos PU (A+T)**
- F3.1 Implementación de bagging PU (Mordelet y Vert, 2014) con Random Forest y Gradient Boosting.
- F3.2 Estimación de la prior de clase π (≥ 2 métodos) y análisis de sensibilidad.
- F3.3 Validación cruzada anidada con todos los esquemas de partición × 10 semillas, para `pre_dft` y `post_dft`, con búsqueda aleatoria de presupuesto fijo y el umbral elegido en el bucle interno.
- F3.4 Validación de **formabilidad** con los negativos de Bartel y las composiciones sintetizadas no perovskita (τ marcado como *in-sample*).
- F3.5 **Obligatoria:** reevaluación del modelo de grafos de Gu et al.
  - Primero se reproduce su partición aleatoria en un contenedor o entorno aislado con su código y versiones originales; después se aplican las particiones por grupos.
  - Se ejecuta en CPU, en Colab o en la GPU del grupo.
  - **Plazo máximo: 3 semanas.**
  - Plan B: entrenar un CGCNN o ALIGNN propio con los mismos datos.
  - Si ambos fallan, H3 pasa a ser exploratoria (se registra en la bitácora).
- F3.6 Análisis de sensibilidad: sin fases de alta presión ni metaestables y sin casos de estado de oxidación ambiguo.

**Fase 4 — Evaluación final e interpretabilidad (A+T)**
- F4.0 Congelar el preregistro y depositarlo en OSF (requisito para abrir el holdout).
- F4.1 Evaluación única sobre el holdout congelado (`HOLDOUT_UNLOCK=1`).
- F4.2 Comparaciones estadísticas:
  - Nadeau–Bengio para CV repetida;
  - bootstrap jerárquico por grupo para leave-one-group-out;
  - no inferioridad para H2;
  - corrección de Holm.
- F4.3 SHAP sobre el mejor modelo `pre_dft`, con estabilidad entre semillas.
- F4.4 Ranking de perovskitas virtuales con su incertidumbre entre semillas.

**Fase 5 — Extensión de tesis: estabilidad acuosa (T, exploratoria, OE4)**
- F5.1 Cálculo de ΔG_pbx **continuo**:
  - con `PourbaixDiagram`, `MaterialsProjectAqueousCompatibility` y `filter_solids=True`;
  - a pH 7,4, ~6,5 (microambiente tumoral) y ~5 (lisosomal);
  - en una ventana de potencial fisiológica (aprox. −0,2 a +0,4 V vs SHE);
  - con sensibilidad a la concentración de iones (10⁻⁶ y 10⁻⁸ M);
  - con un umbral de referencia de ~0,5 eV/átomo (Singh et al., 2017).
  - Solo para candidatos con una entrada GGA/GGA+U consistente.
- F5.2 Filtro de toxicidad (Pb, Ba²⁺ soluble, Cr⁶⁺, Co, Ni, Cd, Tl) y priorización por frente de Pareto: sintetizabilidad frente a estabilidad acuosa.
- F5.3 Discusión de la lixiviación del sitio A (Sr²⁺, Ba²⁺) y de la precipitación con fosfatos y carbonatos. Opcional: hull del sistema extendido con fosfatos y carbonatos.
- F5.4 Contraste retrospectivo con perovskitas de interés biomédico reportadas en la literatura.

**Fase 6 — Escritura (A / T)**
- F6.1 Figuras y tablas del artículo generadas desde `paper/scripts/`.
- F6.2 Borrador del artículo (estructura en `paper/manuscript_outline.md`) y envío a *Digital Discovery*.
- F6.3 Archivo en Zenodo (código y datos derivados redistribuibles) y declaración de uso de IA.
- F6.4 Capítulos de la tesis (estructura en `thesis/chapter_outline.md`).

Al final del ROADMAP incluye dos secciones:
- **Bitácora de decisiones:** tabla `Fecha | Decisión | Justificación | Tarea relacionada`. Precárgala con las decisiones de la versión 2 de este prompt (sección 7), con fecha 2026-11-12.
- **Resultados clave:** tabla que se irá llenando con las métricas principales (media ± DE entre semillas) y enlace al `summary.csv` correspondiente.

---

# 5. Esquema inicial del artículo y de la tesis

**`paper/manuscript_outline.md`** con esta estructura:
1. Introducción: sintetizabilidad, PU learning, el problema de la evaluación con particiones aleatorias y trabajos relacionados (LOCO-CV, MatFold, SynCoTrain, Desai et al., Gu et al.). Delimitación de la contribución.
2. Datos: fuentes, identificación de perovskitas por conectividad, validación manual, deduplicación, etiquetado por composición (positivo / sintetizada no perovskita / no etiquetado) y versión de la base de datos de MP.
3. Métodos: features y su auditoría (incluida la circularidad de E_hull), bagging PU, prior de clase, esquemas de partición y holdout por grupos, protocolo de semillas, métricas PU, tests estadísticos (Nadeau–Bengio, bootstrap jerárquico, no inferioridad) y preregistro.
4. Resultados: (a) efecto de la partición (H1); (b) modelo composicional frente a τ (H2); (c) modelo de grafos frente a composicional y control por simetría (H3); (d) validación de formabilidad; (e) interpretabilidad; (f) análisis de sensibilidad.
5. Discusión: implicaciones para el cribado de materiales, recomendaciones de evaluación, limitaciones (SCAR, ruido de la etiqueta, sesgo de MP hacia fases ordenadas) y desviaciones del preregistro.
6. Disponibilidad de datos y código (Zenodo, restricciones de licencia de ICSD) y declaración de uso de IA.

Para cada sección, anota qué tareas del ROADMAP la alimentan.

**`thesis/chapter_outline.md`**: incluye los mismos capítulos y además:
- un capítulo con el planteamiento formal: problema, justificación, objetivo general y específicos, hipótesis, alcance, delimitaciones y supuestos;
- un capítulo exploratorio de estabilidad acuosa (OE4, Fase 5);
- una discusión de las aplicaciones biomédicas y sus limitaciones: termodinámica sin cinética, sin medio biológico (fosfatos, proteínas, citrato), lixiviación del sitio A y toxicidad.

---

# 6. Qué hacer en esta sesión

1. Crea la estructura de carpetas y todos los archivos descritos en las secciones 2 a 5, incluido `docs/preregistro.md` como borrador con las hipótesis de la sección 1.2.
2. Inicializa el entorno con **uv**:
   - `uv init --package --python 3.12` (paquete `perovskite_pu`, `requires-python = ">=3.12,<3.13"`);
   - `uv add pymatgen mp-api matminer scikit-learn pandas numpy scipy shap pyyaml python-dotenv matplotlib pyarrow`;
   - `uv add --dev pytest ruff pre-commit poethepoet`;
   - confirma con el usuario antes de instalar dependencias;
   - no instales paquetes de deep learning todavía;
   - si uv no logra resolver o instalar las dependencias, informa del error exacto, propone Poetry como respaldo, espera confirmación y registra la decisión en la bitácora.
3. Escribe los esqueletos de `tests/test_leakage.py` y `tests/test_metrics.py`:
   - tests con fixtures sintéticas que **sí se ejecutan** (no solapamiento de composiciones y grupos en las particiones de juguete, análisis estático de `training/`, métricas PU sobre casos conocidos);
   - tests dependientes de datos en `skipif`.
   - Verifica que `uv run pytest -rs` se ejecute sin errores.
4. Inicializa git, configura `.gitattributes` y `pre-commit install`.
   - **Muestra el diff al usuario y pide confirmación antes del primer commit**, sin trailers de coautoría.
5. Actualiza el ROADMAP: F0.1 a F0.3 en ✅ Completada, con fecha y commit, en un commit aparte (también con confirmación).
6. Termina con un resumen breve:
   - qué se creó;
   - qué falta para empezar la Fase 1 (en particular, la clave `MP_API_KEY` en `.env`, la lectura de literatura de F0.5 y la fijación de δ₁ y δ₂ en el preregistro);
   - cualquier decisión que hayas tenido que tomar.

No descargues datos ni entrenes modelos en esta sesión. Si algo de este mensaje es ambiguo o contradictorio, pregunta antes de decidir.

---

# 7. Registro de cambios respecto a la versión 1

| Cambio | Motivo (experto que lo señaló) |
|---|---|
| Positivo = polimorfo **perovskita** en ICSD; nueva categoría "sintetizada no perovskita" | Con "cualquier polimorfo", la ilmenita o YMnO₃ hexagonal contaban como positivos (Química, ML) |
| Identificación por conectividad + validación manual; exclusión de polianiones; hexagonales aparte | StructureMatcher con un solo prototipo pierde perovskitas distorsionadas o acepta falsos positivos (Química, Química computacional) |
| `group_system` queda como diagnóstico; se añaden `loeo_A`, `family`, `cluster` y `temporal`; se usa MatFold | En ABO₃ cada sistema A–B suele contener una sola composición, así que agrupar por sistema ≈ `random` (los 5) |
| Holdout definido por grupos de elementos B + `train_pool` + `load_holdout()` con desbloqueo | Un holdout aleatorio no prueba extrapolación (ML, Metodología, MLOps) |
| Los negativos de Bartel validan **formabilidad**, no sintetizabilidad; τ se marca *in-sample* | Son compuestos sintetizados no perovskita, y τ se ajustó sobre ellos (Química, Química computacional, ML, Metodología) |
| `post_dft` como cota superior; líneas base de solo E_hull y solo simetría; una sola funcional | Circularidad del hull y fuga por procedencia de la estructura (Química computacional, ML, Química) |
| Hipótesis con δ preregistrados; no inferioridad en H2; bootstrap jerárquico en leave-one-group-out | Nadeau–Bengio no aplica a folds deterministas; "iguala o supera" exige no inferioridad (ML, Metodología) |
| H3 con F3.5 obligatoria, plazo de 3 semanas y plan B | Una hipótesis no puede depender de una tarea opcional (ML, Metodología) |
| H4 pasa a OE4 exploratorio; ΔG_pbx continuo, ventana de potencial, pH 6,5, toxicidad, fosfatos | Un enunciado de existencia no es falsable; faltaban limitaciones químicas (Metodología, Química) |
| Objetivo general y específicos, alcance, delimitaciones, cronograma de 20 meses, preregistro OSF, revista objetivo | Formato de tesis y estrategia de publicación (Metodología) |
| Prior de clase, curvas TPR frente a P(ŷ=1), Lee–Liu solo para ordenar modelos | SCAR no se cumple (ML) |
| uv (Poetry como respaldo), paquete `perovskite_pu`, `.gitattributes`, MANIFEST, CI, pre-commit, tests sintéticos siempre activos, `SeedSequence`, árbol limpio | Reproducibilidad en Windows y tests de fuga que no queden saltados para siempre (MLOps) |
| Versión de la base de datos de MP registrada | `theoretical` y E_hull cambian entre versiones (Química computacional, MLOps) |
| Commits solo tras mostrar el diff y recibir confirmación; sin trailer de coautoría | Preferencias del usuario |
