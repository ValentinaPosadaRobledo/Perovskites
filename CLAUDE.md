# CLAUDE.md — Reglas permanentes del proyecto

Estas reglas se aplican en **todas** las sesiones. Si una instrucción nueva las contradice,
pregunta antes de actuar y registra la decisión en la bitácora del `ROADMAP.md`.
El prompt de inicialización completo (versión 2) está en
`docs/prompt_inicializacion_claude_code.v2.md`.

## Proyecto en una línea

Tesis de maestría + artículo (*Digital Discovery*): evaluación sistemática y preregistrada
de modelos PU learning de sintetizabilidad para perovskitas óxido ABO₃ bajo extrapolación
química, con control de fuga de información y reevaluación del modelo de grafos de Gu et
al. (2022). **No** se propone un algoritmo nuevo.

- **Positivo:** la composición tiene al menos un polimorfo **perovskita** (BO₆ que comparten
  solo vértices) con registro experimental en ICSD (`theoretical == False` en MP, contrastado
  con `database_IDs["icsd"]`).
- **Sintetizada no perovskita:** está en ICSD solo con estructuras no perovskita. No es
  positivo ni no etiquetado; se usa en la validación de **formabilidad**.
- **No etiquetado:** sin registro experimental. **No es negativo:** el problema es PU.
- **Banderas de sensibilidad:** alta presión / metaestables y estados de oxidación ambiguos
  (`configs/data.yaml`). Se conservan, se marcan y se repite el análisis sin ellas.
- Hipótesis H1–H3 y objetivos OE1–OE4: ver `docs/preregistro.md`.

---

## 2.1 Control de data leakage (obligatorio)

1. **Identificar perovskitas por conectividad**, no solo por similitud a un prototipo:
   estados de oxidación (`BVAnalyzer` / `oxi_state_guesses`); `CrystalNN`/ChemEnv con B en
   NC = 6 y A en NC 8–12; octaedros BO₆ que comparten **solo vértices** con ángulos
   B–O–B > ~140°; contraste con `StructureMatcher` (tolerancias en `configs/data.yaml`).
   Excluir polianiones (carbonatos, nitratos, boratos…). Clasificar aparte los politipos
   hexagonales 2H/4H/6H. Validar a mano ≥ 50 casos límite en `docs/validacion_manual.md`.
2. **Deduplicar antes de particionar:** `StructureMatcher` para duplicados estructurales,
   luego agrupar por composición reducida. Las composiciones casi idénticas o dopadas se
   agrupan con su compuesto padre.
3. **Etiqueta a nivel de composición.** Nunca la misma composición en train y test.
4. **Esquemas de partición**, siempre reportados por separado y con la distribución de
   tamaños de grupo y de positivos por grupo:
   - `random`: solo referencia para medir la fuga.
   - `group_system`: GroupKFold por sistema A–B. **Solo diagnóstico** (se espera ≈ `random`;
     demostrarlo con la distribución de tamaños de grupo).
   - `loeo_A`, `loeo_B`: leave-one-element-out. Elementos con < 5 positivos se fusionan por
     familia o se reportan aparte.
   - `family`: leave-one-family-out del elemento B (familias en `configs/data.yaml`).
   - `cluster`: LOCO-CV por k-means sobre MAGPIE (Meredig et al., 2018).
   - `temporal`: si hay año de ICSD.
   - Usar **MatFold** si es compatible; si no, `src/perovskite_pu/splits/` y documentar el
     motivo en la bitácora.
5. **Holdout externo congelado, definido por grupos.**
   - Todas las composiciones cuyo B pertenece a un conjunto reservado de elementos, elegidos
     estratificando por familia hasta ~10–15 % de los positivos.
   - Se crea **una sola vez** con `holdout_seed: 2026`.
   - `data/splits/holdout_frozen.csv` en forma canónica (ordenado, LF, UTF-8). SHA-256 en
     `data/splits/HASHES.json` y en el preregistro.
   - En el mismo paso, `data/processed/train_pool.parquet` **sin** el holdout. Todo el
     entrenamiento y ajuste parte de `train_pool`.
   - Solo se lee con `perovskite_pu.evaluation.holdout.load_holdout()`, que exige
     `HOLDOUT_UNLOCK=1`. **Ningún módulo de `src/perovskite_pu/training/` puede importarla
     ni mencionar el holdout** (ni siquiera en docstrings o cadenas: el test estático los
     revisa; usa comentarios `#` si hace falta advertir algo).
6. **Todo el preprocesamiento dentro de un `sklearn.pipeline.Pipeline`**, ajustado solo con
   el train de cada fold. Si τ se usa como feature, excluir de la evaluación los compuestos
   de Bartel o reajustar τ dentro de cada fold.
7. **CV anidada.** Bucle interno: hiperparámetros con búsqueda aleatoria de presupuesto fijo
   (~10–20 configuraciones, definidas en el YAML) **y umbral de decisión**. Bucle externo:
   para reportar.
8. **Auditoría de features.** Ninguna feature derivada de la etiqueta ni posterior a la
   síntesis. Dos familias reportadas por separado:
   - `pre_dft`: composición y factores geométricos (MAGPIE, t, μ, τ).
   - `post_dft`: + `energy_above_hull` y `formation_energy_per_atom`. **Circularidad:** el
     hull de MP lo definen en parte las fases de ICSD y las hipotéticas suelen ser
     prototipos sin relajar inclinaciones. `post_dft` es una **cota superior informada por el
     hull**, junto a una línea base de solo E_hull. Una única funcional
     (`thermo_types=["GGA_GGA+U"]`); sensibilidad con r2SCAN.
9. **Tests de fuga** en `tests/test_leakage.py`, que deben pasar antes de cualquier
   entrenamiento:
   - **Siempre activos** (fixtures sintéticas): sin solapamiento de composiciones ni de
     grupos; preprocesamiento ajustado solo con train; análisis estático (`ast`) de
     `training/`; audit hook sobre `open` que falla si se abre `holdout_frozen.csv`.
   - **Dependientes de datos:** holdout ∩ train_pool = ∅ y hash canónico = `HASHES.json`.
     Se saltan sin datos, pero **fallan** si `REQUIRE_DATA=1` o si `data/processed/` contiene
     algún archivo distinto de `.gitkeep`.
   - Ejecutar siempre con `pytest -rs` (ya está en `addopts`). **Ninguna tarea a partir de
     F1.7 se cierra con tests de fuga saltados** (usar `uv run poe test-data`).

## 2.2 Semillas y estadística (obligatorio)

- Semillas solo en `configs/seeds.yaml`: `seeds: [0..9]`, `holdout_seed: 2026`.
- **Cada experimento con todas las semillas.** De cada semilla,
  `np.random.SeedSequence(seed).spawn(3)` → `fold_assignment`, `pu_sampling`, `model_init`
  (`perovskite_pu.utils.seeds`). En leave-one-group-out los folds son deterministas: la
  semilla solo afecta al bagging y al modelo, y así se declara.
- **Desarrollo incremental:** 3 semillas para medir tiempos (`dev_mode: true`); las 10 son
  obligatorias antes de marcar una tarea como completada.
- **Resultados:** `results/<experimento>/<timestamp>_<hash_config>/` con
  `raw/seed_<n>.csv` (por semilla y fold) y `run_metadata.json`; agregado en
  `results/<experimento>/summary.csv` (media, DE, IC 95 %); índice en `results/runs_index.csv`.
- **Tests estadísticos:** CV repetida con folds aleatorios (`random`, `cluster`) →
  Nadeau–Bengio corregido (nunca t pareado simple). Leave-one-group-out → **bootstrap
  jerárquico por grupo** o permutación por grupo. H2 → **no inferioridad** con margen
  preregistrado. Comparaciones múltiples → Holm.
- **`run_metadata.json`** (`perovskite_pu.utils.run_metadata`): commit; **se niega a correr
  con el árbol sucio** salvo `--allow-dirty` (guarda el diff); fecha; config completa y su
  hash; versiones de paquetes y hash de `uv.lock`; semillas; versión de la BD de MP; CPU,
  `n_jobs` y tiempos.
- **Paralelización:** joblib (`loky`) solo en el nivel semilla × fold; `n_jobs=1` en RF/GB;
  `OMP_NUM_THREADS=1`. En Windows, todo punto de entrada usa `if __name__ == "__main__":`.

## 2.3 Métricas PU

- **Principal:** TPR sobre positivos reservados en un punto de operación fijo.
  **Preregistrado (2026-11-13):** umbral calibrado en el bucle interno a recall nominal
  0,90 (`threshold_for_recall` sobre puntuaciones fuera de fold) y aplicado sin cambios al
  test externo. Las comparaciones entre modelos (H2, H3) se hacen a igual P(ŷ = 1) en cada
  fold. δ₁ = δ₂ = 0,05. Detalle en `docs/preregistro.md`, secciones 7–9.
- **Control:** P(ŷ = 1 | U).
- **Curvas:** TPR frente a P(ŷ = 1) para todos los modelos y líneas base; TPR a varias
  P(ŷ = 1) fijas (para comparar con τ, cuyo umbral 4,18 viene de fuera).
- **Lee y Liu (2003)**, recall²/P(ŷ = 1): supone SCAR, que **no se cumple** → solo para
  **ordenar** modelos.
- **Prior de clase π:** ≥ 2 métodos (Elkan–Noto, TIcE o KM2) + sensibilidad.
- **Validación de formabilidad (no de sintetizabilidad):** negativos de Bartel et al. (2019)
  + "sintetizadas no perovskita". Comprobar que no se solapan con positivos. τ sobre los
  datos de Bartel se marca *in-sample*.
- **Cotas inferiores:** accuracy, F1 y ROC-AUC con U como negativo solo como cotas inferiores
  (las funciones devuelven claves con sufijo `_lower_bound`).
- **`loeo_A` / `loeo_B`:** TPR micro-agregado + distribución por elemento.

## 2.4 Reproducibilidad, entorno y estilo

- **Entorno:** Python 3.12 (`>=3.12,<3.13`) con **uv**. Se versionan `uv.lock` y
  `.python-version`. Todo con `uv run`. Poetry solo como respaldo si uv falla (registrar el
  error exacto y la decisión en la bitácora antes de cambiar).
- **Paquete:** `src/perovskite_pu/`. Tareas con poethepoet: `uv run poe test | test-data |
  lint | format | check | figures | roadmap`.
- **Calidad:** ruff (lint + formato); pre-commit (ruff, detect-secrets, nbstripout,
  check-added-large-files); CI en GitHub Actions (Ubuntu + Windows) con ruff y `pytest -rs`.
- **Secretos:** `MP_API_KEY` en `.env` (nunca versionado); plantilla en `.env.example`.
- **Datos:** crudos en `data/raw/` (no versionados; se versiona el script de descarga).
  `data/MANIFEST.json`: SHA-256 de cada archivo crudo, versión de la BD de MP
  (`MPRester().get_database_version()`), versiones de `mp-api` y `emmet-core`, fecha.
  `data/splits/` sí se versiona (solo composiciones). `.gitattributes`: `*.csv -text`,
  `*.parquet binary`.
- **Licencia ICSD:** las estructuras de ICSD **no se redistribuyen**; solo identificadores y
  features derivadas.
- **Configuración:** todo experimento se lanza desde un YAML en `configs/experiments/`
  (plantilla `_template.yaml`); nada de parámetros escritos en el código.
- **Figuras:** cada figura/tabla del artículo con un script en `paper/scripts/` a partir de
  `results/`; nada se edita a mano; cada figura con un `.json` (hashes de los `summary.csv`
  de entrada + commit). `uv run poe figures` regenera todas.
- **Idioma** *(modificado el 2026-11-13 a petición de la autora; ver bitácora)*:
  - Identificadores (variables, funciones, clases, módulos) en **inglés**.
  - **Comentarios y docstrings en español.** Todo el código debe ir comentado.
  - Documentos del proyecto (ROADMAP, outlines, bitácora) y mensajes de commit en español.
- **Git:**
  - Commits pequeños con mensajes descriptivos en español.
  - **Antes de cada commit, mostrar el diff y pedir confirmación explícita a la usuaria.**
  - **No añadir trailers de coautoría de IA** (ni `Co-Authored-By` ni enlaces de sesión).
  - Autoría: Valentina Posada.
- **Publicación:** al cerrar, Zenodo (código + datos derivados redistribuibles) con DOI.
  Declarar el uso de asistentes de IA (Claude Code) según la política de la revista; la IA
  no figura como autora.

## 2.5 Mantenimiento del ROADMAP

- Al empezar una tarea: **🔄 En proceso** + fecha de inicio (`uv run poe roadmap start <ID>`).
- Al terminarla: **✅ Completada** + fecha + commit **que produjo el resultado** + enlace al
  resultado (`uv run poe roadmap done <ID> --result "..."`). La actualización del ROADMAP va
  en un **commit aparte**.
- Si queda bloqueada: sigue en 🔄 con el motivo en notas (`roadmap block <ID> --note "..."`).
- Nunca marcar como completada una tarea con tests fallidos o saltados, o sin resultados de
  todas las semillas.
- Toda decisión metodológica relevante → "Bitácora de decisiones" del ROADMAP (fecha y
  justificación). Un resultado nulo también es un resultado y se registra.

## 2.6 Preregistro

`docs/preregistro.md`: definición de positivo, esquemas de partición, holdout y su hash,
hipótesis con δ₁ y δ₂, tests estadísticos, punto de operación y análisis de sensibilidad.
Se congela y se deposita en OSF **antes de F4** (F4.0 es requisito para abrir el holdout).
Toda desviación posterior se declara en el artículo.

---

## Forma de trabajo con la autora

- Ir por fases; reportar avances **y resultados, incluidos los nulos**.
- Preguntar ante cualquier ambigüedad o contradicción antes de decidir.
- Confirmar antes de instalar dependencias, descargar datos o lanzar entrenamientos largos.
