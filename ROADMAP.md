# ROADMAP

**Proyecto:** Evaluación de la generalización de modelos de aprendizaje positivo–no
etiquetado (PU learning) para predecir la sintetizabilidad de perovskitas óxido ABO₃ con
control de fuga de información.

**Leyenda de estados:** ⬜ Pendiente · 🔄 En proceso · ✅ Completada
**Destino:** A = artículo · T = tesis · A+T = ambos

Reglas de mantenimiento: `CLAUDE.md`, sección 2.5. Herramienta:
`uv run poe roadmap start|done|block <ID>`.

## Cronograma de referencia (~20 meses)

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

**Núcleo mínimo publicable:** Fases 1–4 con H1, H2 y la reevaluación de Gu et al. bajo
partición por grupos. SHAP con estabilidad entre semillas, la partición temporal y la Fase 5
son ampliaciones.

---

## Fase 0 — Inicialización (A+T)

| ID | Tarea | Destino | Estado | Inicio | Fin | Commit | Resultado / Notas |
|---|---|---|---|---|---|---|---|
| F0.1 | Estructura del repositorio, CLAUDE.md, ROADMAP.md, README.md, entorno uv (o Poetry como respaldo), `.gitattributes`, pre-commit y CI | A+T | ✅ Completada | 2026-11-13 | 2026-11-13 | `c00fd27` | Estructura, CLAUDE.md, ROADMAP.md, README.md, uv.lock, pre-commit y CI (`.github/workflows/ci.yml`). uv sin conflictos: no se usó Poetry |
| F0.2 | Esqueleto de `tests/test_leakage.py` y `tests/test_metrics.py`, con fixtures sintéticas activas y tests dependientes de datos en `skipif` | A+T | ✅ Completada | 2026-11-13 | 2026-11-13 | `c00fd27` | `tests/test_leakage.py`, `tests/test_metrics.py`, `tests/test_utils.py`: 96 pasan, 2 saltados (dependientes de datos, hasta F1.8). Encontró y se corrigió un bug de validación en las métricas |
| F0.3 | Configuración de semillas (`SeedSequence`) y plantilla de configuración de experimentos | A+T | ✅ Completada | 2026-11-13 | 2026-11-13 | `c00fd27` | `configs/seeds.yaml`, `src/perovskite_pu/utils/seeds.py` (SeedSequence → 3 flujos), `configs/experiments/_template.yaml` |
| F0.4 | Borrador de `docs/preregistro.md` (hipótesis, δ₁, δ₂, tests, punto de operación) | A+T | ✅ Completada | 2026-11-13 | 2026-11-13 | `e9ff6bc` | `docs/preregistro.md` §7–9: δ₁ = δ₂ = 0,05; punto de operación en recall 0,90 calibrado en el bucle interno; reglas de decisión H1–H3; potencia en `scripts/potencia_preregistro.py`. Sigue en borrador hasta F4.0 |
| F0.5 | Revisión de literatura: verificar y leer MatFold, LOCO-CV, SynCoTrain, Desai et al. (2025), Gu et al. (2022), Bartel et al. (2019), Jang et al. (2020), Frey et al. (2019) y SynthNN. Delimitar la brecha | A+T | 🔄 En proceso | 2026-11-13 | | | **Bloqueada (2026-11-13):** Revisión parcial en `docs/revision_literatura.md` (22 refs verificadas, 5 leídas completas). Falta leer a texto completo Chung 2025 (prioridad), MatFold, Meredig y Bartel: la RSC bloquea el acceso automatizado. Faltan además Lee y Liu 2003 y KM2 |

## Fase 1 — Datos (A+T)

| ID | Tarea | Destino | Estado | Inicio | Fin | Commit | Resultado / Notas |
|---|---|---|---|---|---|---|---|
| F1.1 | Descarga desde MP con `mp-api` (`material_id`, `formula_pretty`, `structure`, `theoretical`, `database_IDs`, `energy_above_hull`, `formation_energy_per_atom`, `symmetry`), `thermo_types=["GGA_GGA+U"]`, versión de la BD en `data/MANIFEST.json` | A+T | ✅ Completada | 2026-11-13 | 2026-11-13 | `6218285` | `data/MANIFEST.json`: MP 2026.09.28; 2563 entradas ABO₃ (1319 composiciones; 829 experimentales; 0 GNoME); E_hull GGA/GGA+U para 2490. Descarga reproducible (hashes idénticos en 3 corridas) |
| F1.2 | Descarga del dataset de Gu et al. (Zenodo) y de Bartel et al. (2019); verificación de su solapamiento con MP | A+T | ✅ Completada | 2026-11-13 | 2026-11-13 | `13fbe30` | `results/f1_2_datos_externos/informe.md`: Gu 12 907 cristales (7502 ABO₃, 312 positivos); fuga entre polimorfos del 28 % en su test; Bartel 369 óxidos ABO₃; 16 conflictos Bartel–Gu; 34 perovskitas de Bartel solo teóricas en MP |
| F1.3 | Identificación de perovskitas ABO₃ por conectividad (CrystalNN/ChemEnv + StructureMatcher, regla 2.1.1): exclusión de polianiones, politipos hexagonales aparte, estados de oxidación y casos ambiguos | A+T | ⬜ Pendiente | | | | |
| F1.4 | Validación manual de ≥ 50 casos límite en `docs/validacion_manual.md` | A+T | ⬜ Pendiente | | | | |
| F1.5 | Deduplicación estructural y por composición, agrupación de casi duplicados y etiquetado por composición (positivo / sintetizada no perovskita / no etiquetado), con banderas de alta presión y metaestabilidad | A+T | ⬜ Pendiente | | | | |
| F1.6 | Informe de conteos (positivos, sintetizadas no perovskita, no etiquetados; por estructura, composición, elemento A y B y familia). **Decisión:** si hay < ~100 ABO₃ positivas (estimación previa 200–350), ampliar a todas las clases de perovskita y evaluar óxidos aparte | A+T | ⬜ Pendiente | | | | Registrar la decisión en la bitácora |
| F1.7 | Generación de todas las particiones (MatFold o propia) con informe de tamaños de grupo y positivos por grupo; demostrar cuánto se parece `group_system` a `random`. **S1:** distancia al train por grupo y fracción de grupos extrapolativos por esquema (preregistro §10b; requiere F2.1) | A+T | ⬜ Pendiente | | | | Implementación provisional propia ya en `src/perovskite_pu/splits/` |
| F1.8 | Holdout congelado por grupos (y temporal si aplica), `train_pool.parquet` y hash canónico en `HASHES.json`. Todos los tests de fuga pasan sin saltos | A+T | ⬜ Pendiente | | | | |

## Fase 2 — Features y líneas base (A+T)

| ID | Tarea | Destino | Estado | Inicio | Fin | Commit | Resultado / Notas |
|---|---|---|---|---|---|---|---|
| F2.1 | Features `pre_dft`: MAGPIE (ElementProperty, Stoichiometry, ValenceOrbital, IonProperty) + t, μ, τ (Shannon, A en XII y B en VI; radios extrapolados documentados) | A+T | ⬜ Pendiente | | | | |
| F2.2 | Features `post_dft`: `pre_dft` + E_hull y ΔHf (una sola funcional) | A+T | ⬜ Pendiente | | | | |
| F2.3 | Auditoría de features documentada, incluida la circularidad de E_hull | A+T | ⬜ Pendiente | | | | |
| F2.4 | Líneas base con todos los esquemas y semillas: t de Goldschmidt, τ de Bartel, umbral de E_hull, modelo solo E_hull, modelo solo grupo espacial/simetría, clasificador trivial | A+T | ⬜ Pendiente | | | | |

## Fase 3 — Modelos PU (A+T)

| ID | Tarea | Destino | Estado | Inicio | Fin | Commit | Resultado / Notas |
|---|---|---|---|---|---|---|---|
| F3.1 | Bagging PU (Mordelet y Vert, 2014) con Random Forest y Gradient Boosting | A+T | ⬜ Pendiente | | | | |
| F3.2 | Estimación de la prior de clase π (≥ 2 métodos) y análisis de sensibilidad | A+T | ⬜ Pendiente | | | | |
| F3.3 | CV anidada: todos los esquemas × 10 semillas, `pre_dft` y `post_dft`, búsqueda aleatoria de presupuesto fijo, umbral en el bucle interno | A+T | ⬜ Pendiente | | | | |
| F3.4 | Validación de **formabilidad** con negativos de Bartel y sintetizadas no perovskita (τ marcado *in-sample*) | A+T | ⬜ Pendiente | | | | |
| F3.5 | **Obligatoria:** reevaluación del modelo de grafos de Gu et al.: reproducir su partición aleatoria en entorno aislado con su código y versiones; **S2:** repetirla agrupando sus cristales por composición (ΔTPR por fuga entre polimorfos); luego particiones por grupos. CPU/Colab/GPU del grupo. **Plazo máximo: 3 semanas.** Plan B: CGCNN o ALIGNN propio. Si ambos fallan, H3 pasa a exploratoria (bitácora) | A | ⬜ Pendiente | | | | |
| F3.6 | Sensibilidad: sin fases de alta presión/metaestables y sin estados de oxidación ambiguos | A+T | ⬜ Pendiente | | | | |

## Fase 4 — Evaluación final e interpretabilidad (A+T)

| ID | Tarea | Destino | Estado | Inicio | Fin | Commit | Resultado / Notas |
|---|---|---|---|---|---|---|---|
| F4.0 | Congelar el preregistro y depositarlo en OSF (requisito para abrir el holdout) | A+T | ⬜ Pendiente | | | | |
| F4.1 | Evaluación única sobre el holdout congelado (`HOLDOUT_UNLOCK=1`) | A+T | ⬜ Pendiente | | | | |
| F4.2 | Comparaciones estadísticas: Nadeau–Bengio (CV repetida), bootstrap jerárquico por grupo (leave-one-group-out), no inferioridad (H2), Holm | A+T | ⬜ Pendiente | | | | |
| F4.3 | SHAP sobre el mejor modelo `pre_dft`, con estabilidad entre semillas | A+T | ⬜ Pendiente | | | | Ampliación (fuera del núcleo mínimo) |
| F4.4 | Ranking de perovskitas virtuales con incertidumbre entre semillas | A+T | ⬜ Pendiente | | | | |

## Fase 5 — Extensión de tesis: estabilidad acuosa (T, exploratoria, OE4)

| ID | Tarea | Destino | Estado | Inicio | Fin | Commit | Resultado / Notas |
|---|---|---|---|---|---|---|---|
| F5.1 | ΔG_pbx **continuo** con `PourbaixDiagram`, `MaterialsProjectAqueousCompatibility`, `filter_solids=True`; pH 7,4, ~6,5 y ~5; ventana −0,2 a +0,4 V vs SHE; iones 10⁻⁶ y 10⁻⁸ M; referencia ~0,5 eV/átomo (Singh et al., 2017); solo candidatos con entrada GGA/GGA+U consistente | T | ⬜ Pendiente | | | | |
| F5.2 | Filtro de toxicidad (Pb, Ba²⁺ soluble, Cr⁶⁺, Co, Ni, Cd, Tl) y frente de Pareto sintetizabilidad–estabilidad acuosa | T | ⬜ Pendiente | | | | |
| F5.3 | Discusión de lixiviación del sitio A (Sr²⁺, Ba²⁺) y precipitación con fosfatos/carbonatos. Opcional: hull extendido con fosfatos y carbonatos | T | ⬜ Pendiente | | | | |
| F5.4 | Contraste retrospectivo con perovskitas de interés biomédico reportadas | T | ⬜ Pendiente | | | | |

## Fase 6 — Escritura (A / T)

| ID | Tarea | Destino | Estado | Inicio | Fin | Commit | Resultado / Notas |
|---|---|---|---|---|---|---|---|
| F6.1 | Figuras y tablas del artículo generadas desde `paper/scripts/` | A | ⬜ Pendiente | | | | |
| F6.2 | Borrador del artículo (`paper/manuscript_outline.md`) y envío a *Digital Discovery* | A | ⬜ Pendiente | | | | |
| F6.3 | Archivo en Zenodo (código y datos derivados redistribuibles) y declaración de uso de IA | A+T | ⬜ Pendiente | | | | |
| F6.4 | Capítulos de la tesis (`thesis/chapter_outline.md`) | T | ⬜ Pendiente | | | | |

---

## Bitácora de decisiones

| Fecha | Decisión | Justificación | Tarea relacionada |
|---|---|---|---|
| 2026-11-12 | Positivo = polimorfo **perovskita** en ICSD; nueva categoría "sintetizada no perovskita" | Con "cualquier polimorfo", la ilmenita o YMnO₃ hexagonal contaban como positivos (Química, ML) | F1.5 |
| 2026-11-12 | Identificación por conectividad + validación manual; exclusión de polianiones; hexagonales aparte | StructureMatcher con un solo prototipo pierde perovskitas distorsionadas o acepta falsos positivos (Química, Química computacional) | F1.3, F1.4 |
| 2026-11-12 | `group_system` solo como diagnóstico; se añaden `loeo_A`, `family`, `cluster` y `temporal`; se usa MatFold | En ABO₃ cada sistema A–B suele contener una sola composición, así que agrupar por sistema ≈ `random` (los 5 especialistas) | F1.7 |
| 2026-11-12 | Holdout definido por grupos de elementos B + `train_pool` + `load_holdout()` con desbloqueo | Un holdout aleatorio no prueba extrapolación (ML, Metodología, MLOps) | F1.8, F4.1 |
| 2026-11-12 | Los negativos de Bartel validan **formabilidad**, no sintetizabilidad; τ se marca *in-sample* | Son compuestos sintetizados no perovskita, y τ se ajustó sobre ellos (Química, Química computacional, ML, Metodología) | F3.4 |
| 2026-11-12 | `post_dft` como cota superior; líneas base de solo E_hull y solo simetría; una sola funcional | Circularidad del hull y fuga por procedencia de la estructura (Química computacional, ML, Química) | F2.2–F2.4 |
| 2026-11-12 | Hipótesis con δ preregistrados; no inferioridad en H2; bootstrap jerárquico en leave-one-group-out | Nadeau–Bengio no aplica a folds deterministas; "iguala o supera" exige no inferioridad (ML, Metodología) | F0.4, F4.2 |
| 2026-11-12 | H3 con F3.5 obligatoria, plazo de 3 semanas y plan B | Una hipótesis no puede depender de una tarea opcional (ML, Metodología) | F3.5 |
| 2026-11-12 | H4 pasa a OE4 exploratorio; ΔG_pbx continuo, ventana de potencial, pH 6,5, toxicidad, fosfatos | Un enunciado de existencia no es falsable; faltaban limitaciones químicas (Metodología, Química) | F5.1–F5.4 |
| 2026-11-12 | Objetivo general y específicos, alcance, delimitaciones, cronograma de 20 meses, preregistro OSF, revista objetivo | Formato de tesis y estrategia de publicación (Metodología) | F0.4, F6 |
| 2026-11-12 | Prior de clase, curvas TPR frente a P(ŷ=1), Lee–Liu solo para ordenar modelos | SCAR no se cumple (ML) | F3.2 |
| 2026-11-12 | uv (Poetry como respaldo), paquete `perovskite_pu`, `.gitattributes`, MANIFEST, CI, pre-commit, tests sintéticos siempre activos, `SeedSequence`, árbol limpio | Reproducibilidad en Windows y tests de fuga que no queden saltados para siempre (MLOps) | F0.1–F0.3 |
| 2026-11-12 | Versión de la base de datos de MP registrada | `theoretical` y E_hull cambian entre versiones (Química computacional, MLOps) | F1.1 |
| 2026-11-12 | Commits solo tras mostrar el diff y recibir confirmación; sin trailer de coautoría | Preferencias de la autora | Todas |
| 2026-11-13 | uv resolvió e instaló todas las dependencias sin conflictos; **no se usa Poetry**. Versiones clave: pymatgen 2026.10.24, mp-api 0.47.2, emmet-core 0.88.3, matminer 0.10.1, scikit-learn 1.9.1, numpy 2.5.4, pandas 2.3.3, shap 0.54.1 | El respaldo solo aplica si uv falla. pandas queda en 2.3.x porque matminer 0.10.1 exige `pandas<3` (verificado en los metadatos del paquete) | F0.1 |
| 2026-11-13 | **Identificadores en inglés; comentarios y docstrings en español** (modifica la regla 2.4, que pedía comentarios en inglés) | Petición explícita de la autora: todo el código comentado en español. Los identificadores siguen en inglés por coherencia con pymatgen/sklearn | F0.1 |
| 2026-11-13 | `data/splits/` (incluido `holdout_frozen.csv`) **sí se versiona** en git | Solo contiene composiciones (nada licenciado por ICSD); versionarlo hace el holdout trazable e inmutable junto a su hash | F1.8 |
| 2026-11-13 | "Existe `data/processed/`" (regla 2.1.9) se interpreta como "contiene algún archivo distinto de `.gitkeep`" | El directorio existe siempre en el repo; con la lectura literal los tests de datos fallarían desde el primer día | F0.2 |
| 2026-11-13 | El análisis estático de `training/` revisa imports, nombres, atributos, argumentos y **todas** las cadenas, docstrings incluidos | Detecta también `importlib.import_module("...holdout")` y rutas en texto. Consecuencia: en `training/` las advertencias sobre el holdout van en comentarios `#` | F0.2 |
| 2026-11-13 | El audit hook sobre `open` complementa al análisis estático, pero no lo sustituye | Solo intercepta aperturas hechas desde Python; una lectura en C pura (p. ej. pyarrow nativo) no dispara el evento | F0.2 |
| 2026-11-13 | Particiones implementadas provisionalmente en `src/perovskite_pu/splits/` (los 7 esquemas, con verificación de fuga en tiempo de ejecución). MatFold se evaluará en F1.7 | Hacían falta para los tests sintéticos de F0.2; la decisión MatFold vs. propia requiere los datos reales | F1.7 |
| 2026-11-13 | Familias de B (propuesta): Sc en 3d, Y en 4d/5d, La en 4f (serie La–Lu); el resto de no listados en `main_group` | Asignación convencional por bloque/periodo; se revisa con los conteos de F1.6 | F1.6, F1.7 |
| 2026-11-13 | CI en Ubuntu **y Windows**; `astral-sh/setup-uv` fijado a `v10.3.3` (no publica etiqueta `v10`) y `actions/checkout@v7` | La autora puede trabajar en Windows (CRLF, `if __name__`); versiones verificadas con `git ls-remote` | F0.1 |
| 2026-11-13 | Se guarda una copia del prompt v2 en `docs/prompt_inicializacion_claude_code.v2.md`. La v1 citada en el prompt no se recibió y no está en el repo | Trazabilidad de las decisiones de inicialización | F0.1 |
| 2026-11-13 | Bug encontrado por los tests de F0.2 y corregido: la validación de entradas binarias de las métricas convertía a entero **antes** de validar (0,3 pasaba como 0) | Ahora valida antes de convertir; `test_rejects_non_binary_inputs` lo cubre | F0.2 |
| 2026-11-13 | **δ₁ = 0,05** (TPR absoluta): caída mínima relevante al pasar de `random` a grupos | 1 de cada 20 perovskitas conocidas; la potencia es buena para caídas ≥ 0,10 (0,71–0,95 con N = 250; `scripts/potencia_preregistro.py`). Fijado antes de descargar datos | F0.4 |
| 2026-11-13 | **δ₂ = 0,05** en lugar del 0,03 propuesto en el prompt v2 | Con 0,03, H2 sería casi siempre inconclusa aunque el modelo PU fuera igual a τ (potencia 0,15–0,32 con N = 250); con 0,05 sube a 0,33–0,71. Además usa la misma diferencia mínima relevante que δ₁. Un resultado inconclusivo se reportará como tal | F0.4 |
| 2026-11-13 | **Punto de operación principal: recall nominal 0,90** calibrado en el bucle interno (percentil 10 de las puntuaciones fuera de fold de los positivos), aplicado sin cambios al test externo. Reemplaza la propuesta del borrador (P(ŷ=1) = 0,10) | No depende de la fracción de positivos ni de π, que aún se desconocen; con P(ŷ=1) = 0,10 la TPR podría quedar limitada artificialmente. Sensibilidad: 0,80 y 0,95 | F0.4, F3.3 |
| 2026-11-13 | Comparaciones entre modelos a **igual P(ŷ=1) en cada fold**: H2 a la P(ŷ=1) que produce τ < 4,18; H3 a la del PU `pre_dft` en su punto de operación | El umbral de τ viene de fuera y el del grafo no se calibra igual; igualar P(ŷ=1) usa solo la distribución de puntuaciones, no las etiquetas | F0.4, F3.5, F4.2 |
| 2026-11-13 | El aprendiz base (RF/GB) es un hiperparámetro del bucle interno; selección por mínima P(ŷ=1) a recall 0,90 | Así "el mejor modelo PU `pre_dft`" lo define el protocolo y no se elige mirando resultados externos | F0.4, F3.3 |
| 2026-11-13 | Tests confirmatorios (H1–H3): bootstrap jerárquico pareado por composición, B = 10 000. Unidades de remuestreo: elemento A en `loeo_A`; elemento B en `loeo_B`, `cluster` y `family` (en este último, estratificado por familia). Holm sobre los 4 esquemas de cada hipótesis; conclusión global por intersección-unión. `bootstrap_seed: 20261113` | Con solo 4 familias no se pueden remuestrear las familias mismas. Nadeau–Bengio queda para comparaciones secundarias dentro de `random` y `cluster` | F0.4, F4.2 |
| 2026-11-13 | H2: el análisis principal incluye los compuestos de Bartel (τ *in-sample*); sensibilidad sin ellos | Incluirlos favorece a τ, así que es conservador para el modelo PU | F0.4, F3.4 |
| 2026-11-13 | H3: regla de interpretación preregistrada. Si en `random` la línea base de solo simetría alcanza ≥ 50 % de la ventaja del grafo, esa ventaja se atribuye posiblemente a la procedencia de la estructura | Hace operativo el control obligatorio de H3 antes de ver resultados | F0.4, F3.5 |
| 2026-11-13 | Entorno de nube propio **perovskites** (personal), con acceso a arxiv.org, export.arxiv.org, doi.org, www.nature.com, idp.nature.com, pubs.rsc.org, zenodo.org, api.crossref.org, api.openalex.org, api.semanticscholar.org y api.materialsproject.org, más un script de arranque (`uv sync --locked` y `pre-commit install`) | El entorno Default no permitía llegar a las fuentes bibliográficas; se creó uno propio para no abrir el que usan todos los proyectos. El script de arranque solo queda probado cuando arranque una sesión nueva | F0.5, F1.1 |
| 2026-11-13 | Correcciones bibliográficas: Frey et al. es de **2019** (no 2020); SynCoTrain y Desai et al. ya están publicados (*Digital Discovery* 2025; *MLST* 2025); Desai et al. cubre solo perovskitas haluro y calcogenuro | Verificado contra Crossref y arXiv (`docs/revision_literatura.md` §2) | F0.5 |
| 2026-11-13 | Gu et al. (2022) evalúa con **una sola partición aleatoria del 10 %, por cristal**, y compara con τ en subconjuntos distintos (310 frente a 943 cristales), no a igual P(ŷ=1) | Leído a texto completo. Refuerza el diseño de H2 (igual P(ŷ=1)) y de F3.5. Se propone medir aparte el efecto de deduplicar por composición | F0.5, F3.5 |
| 2026-11-13 | MatFold v1.3.3, **compatibilidad parcial (preliminar)**: cubre `random`, `chemsys` y `sgnum`, pero no `loeo_A`/`loeo_B` por sitio, `family`, `cluster` ni `temporal`, y exige una estructura por entrada. Se propone un esquema híbrido | Revisado en el README oficial (PyPI). La decisión final queda para F1.7 | F1.7 |
| 2026-11-13 | La brecha queda delimitada (`docs/revision_literatura.md` §6) pero **condicionada a leer Chung, Walsh y Payne (2025)**, PU para óxidos ternarios, cuyo protocolo de evaluación se desconoce | La RSC bloquea el acceso automatizado; hace falta lectura humana o el PDF subido a la sesión | F0.5 |
| 2026-11-13 | **Aprobado por la autora:** análisis secundario **S1** (¿los esquemas por grupos extrapolan de verdad?, por Li et al. 2025) y **S2** (efecto de deduplicar por composición la partición de Gu et al.). Ambos preregistrados como no confirmatorios (§10b) | S1 se evalúa en F1.7, antes de entrenar; S2 en F3.5 con los datos de Gu. Código de S1 en `perovskite_pu.evaluation.extrapolation`, con tests | F1.7, F3.5 |
| 2026-11-13 | S1 usa PCA al 95 % de la varianza (ajustado en train) como espacio principal; el espacio estandarizado completo queda como sensibilidad | Los tests sintéticos mostraron que, al estandarizar, una sola feature de ruido diluye la señal de extrapolación. Con ~130 features MAGPIE el efecto sería mayor | F1.7 |
| 2026-11-13 | La clave de MP se guarda como **secreto de red** del entorno de nube (cabecera `X-API-Key` en api.materialsproject.org) y no en `.env`. El código recibe un valor ficticio de 32 caracteres, que el proxy reemplaza | En la nube, la clave nunca queda en el contenedor ni en el chat. En una instalación local (p. ej. Windows) sigue valiendo `.env` con `MP_API_KEY` (regla 2.4) | F1.1 |
| 2026-11-13 | Se excluyen las entradas de MP marcadas como obsoletas (`deprecated=False` en la consulta) | MP las retira por errores; no son datos válidos | F1.1 |
| 2026-11-13 | **GNoME: fuera del análisis principal, marcado (`is_gnome`) para sensibilidad** (decisión de la autora). **Resultado nulo:** en MP 2026.09.28 no hay ninguna entrada GNoME entre las ABC3 con O. Confirmado con dos métodos: las consultas con y sin GNoME devuelven los mismos 2688 IDs, y ninguno de los 21 lotes (`builder_meta.batch_id`) es de GNoME | Las estructuras de GNoME, generadas por un modelo y seleccionadas por estabilidad, introducirían sesgo de procedencia en U. Gu et al. usó MP sin GNoME | F1.1 |
| 2026-11-13 | La E_hull y la ΔHf del análisis salen del endpoint `thermo` con `GGA_GGA+U` explícito (r2SCAN aparte, para sensibilidad). La E_hull de `summary` se guarda como `summary_energy_above_hull` y no se usa | La E_hull de `summary` usa el esquema mixto GGA/GGA+U/r2SCAN, lo que violaría la regla de una sola funcional | F1.1, F2.2 |
| 2026-11-13 | Descarga reproducible: los documentos se ordenan por `material_id` antes de guardarlos | La primera versión guardaba en el orden de la API y el hash de `summary.json.gz` cambió entre corridas. Con el orden fijo, tres descargas dieron hashes idénticos | F1.1 |
| 2026-11-13 | Primeros hallazgos para F1.5: 55 entradas con `theoretical=True` pero con ID de ICSD; 3 composiciones con ICSD y sin ninguna entrada experimental; 73 entradas (2,8 %) sin termodinámica GGA/GGA+U; 445 de 1319 composiciones con alguna entrada experimental, que es el techo de positivos antes de exigir estructura perovskita | Se resuelven en F1.5 con la regla de positivo (`theoretical == False` contrastado con ICSD). Las 73 sin GGA solo sirven para `pre_dft` | F1.5, F2.2 |
| 2026-11-13 | **MP 2026 cambió el formato de sus IDs** (`mp-5229` → `mp-aaaaahtd`). Se convierten de forma determinista (base 26, `emmet.core.mpid.AlphaID`) con `to_legacy_mpid` | Sin la conversión, 0 de 1182 IDs de Gu se encontraban en MP; con ella, 1167 (todos con la misma composición). Afecta a cualquier cruce con la literatura | F1.2 |
| 2026-11-13 | El repositorio de Gu usa **dos índices**: el de cristal (`#` en `prediction.csv`, el de las particiones) y el de archivo CIF (`data/`). Se enlazan por el conjunto de fuentes, que es único: 12 907 de 12 907, con elementos y etiquetas 100 % coherentes | Cruzar por posición mezclaría etiquetas (1862 discrepancias aparentes) | F1.2, F3.5 |
| 2026-11-13 | Del zip de Gu solo se extraen `.md`, `.csv`, `.json` y `.cif`. Los pesos (`.pkl`, `.pth.tar`) no se abren fuera del entorno aislado de F3.5 | Cargar un pickle ajeno puede ejecutar código | F1.2, F3.5 |
| 2026-11-13 | **Fuga entre polimorfos en el test de Gu:** el 28 % de sus positivos de test (37 % en óxidos ABO₃) tiene su composición también en train/val, y el 22 % (20 %) entre los positivos de train/val. Su test es el mismo en las 100 particiones | Confirma que S2 (preregistro §10b) es necesario | F1.2, F3.5 |
| 2026-11-13 | **16 conflictos Bartel–Gu:** compuestos sintetizados NO perovskita para Bartel que Gu cuenta como positivos (polianiones como CdCO₃, selenitos, teluritos, bromatos, cloratos y SrSiO₃, e ilmenitas FeTiO₃ y MnTiO₃). Además, 41 perovskitas de Bartel figuran en Gu solo como virtuales (p. ej. BaHfO₃) | Coherente con que la identificación por prototipo acepte falsos positivos (regla 2.1.1). Es un conflicto entre fuentes, no un error demostrado: entran como casos límite en F1.4 | F1.3, F1.4, F3.4 |
| 2026-11-13 | **34 perovskitas experimentales de Bartel figuran en MP 2026 solo como teóricas**: con nuestra definición serían no etiquetadas | Medida directa de positivos ocultos en U; contraste para la prior π (F3.2) | F1.5, F3.2 |

## Resultados clave

| Fecha | Experimento | Esquema | Modelo | Métrica (media ± DE, 10 semillas) | `summary.csv` | Commit |
|---|---|---|---|---|---|---|
| | | | | *Aún no hay resultados experimentales (Fase 0).* | | |
