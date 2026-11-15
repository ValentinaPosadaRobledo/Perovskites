# F3.6 — Diseño de los análisis de sensibilidad (borrador para la autora)

Fuente: preregistro §10 (14 sensibilidades) y §10b. Solo el modelo confirmatorio `pu_pre_dft`
salvo que se indique. Coste medido en F3.3: 105 tareas por semilla (`loeo_A` 69, `loeo_B` 16,
`random` 5, `group_system` 5, `cluster` 5, `family` 4, `temporal` 1); ≈ 0,25 min por tarea con
solo `pre_dft` en 4 núcleos → **≈ 4 h por sensibilidad reentrenada con 10 semillas**
(`loeo_A` es el 66 % del coste; quitar `group_system` y `temporal` ahorra solo ~6 %).

## A. Ya hechas (no cuestan nada más)

| § | Sensibilidad | Dónde |
|---|---|---|
| 3 | π con métodos distintos | F3.2 (Elkan–Noto RF/GB, sigmoide; TIcE uniforme/cuantiles) |
| 7 | Formabilidad | F3.4 |
| 9 | H2 sin compuestos de Bartel | F3.3 (descriptivo); prueba formal en F4.2 (`H2_sensitivity`) |

## B. Baratas (minutos; sin reentrenar)

| § | Sensibilidad | Cómo |
|---|---|---|
| 10 | S1 sin PCA | Recalcular las distancias de F1.7 en el espacio estandarizado completo (no se hizo en F1.7) |
| 5b | Tolerancias de StructureMatcher | La deduplicación es solo informativa (bitácora 2026-11-14): no cambia etiquetas. Se declara efecto nulo por construcción y se reportan los recuentos de duplicados con otras tolerancias |

## C. Requieren reentrenar (≈ 4 h cada una salvo que se indique)

| § | Sensibilidad | Efecto en `train_pool` | Nota |
|---|---|---|---|
| 1 | Sin alta presión ni metaestables | quita 11 P (3 + 8) | Ver decisión D1 |
| 2 | Sin estados de oxidación ambiguos | quita 13 P | Ver decisión D1 |
| 5a | Ángulo B–O–B mínimo 140° (en vez de 135°) | ≈ 17 P dejan de serlo (p. ej. GdScO₃, DyScO₃, LuFeO₃: Pnma muy inclinadas); recuento aproximado con los ángulos guardados, hay que reetiquetar con el pipeline | Cambia etiquetas |
| 8 | Recall 0,80 y 0,95 | ninguno | F3.3 solo guardó la P(ŷ=1) y el umbral a 0,90 de cada candidato. Con un cambio en el ejecutor (calcular los 3 recalls en una pasada) cuesta ≈ 4–5 h en total, en vez de ≈ 8 h |
| 11 | Solo U con balance de carga | quita 289 U (quedan 442) | ≈ 3 h. Ver decisión D2 |
| 12 | Sitios A/B por tamaño (r_A > r_B) | 45 composiciones cambian de asignación | Solo `loeo_A` y `loeo_B` (≈ 3,5 h); particiones nuevas en un archivo aparte, sin tocar las congeladas |
| 6 | `loeo_B`: grupos pequeños aparte (no fusionados) | cambia los folds de `loeo_B` | Solo `loeo_B` (≈ 0,7 h) |
| 13 | Ambiguas como P | añade 24 P (las 25 ambiguas menos 1 con B reservado) | Cambia etiquetas |
| 14 | `pre_dft_tau` (+ τ) | ninguno | Se evalúa en todo el conjunto (*in-sample*) y fuera de la TableS1 |
| 4 | r2SCAN en `post_dft` | 173 composiciones (126 P, 47 U) | Descriptiva; conjunto pequeño (≈ 0,5–1 h) |

**Coste total de C:** ≈ 27–29 h de cómputo en 4 núcleos (≈ 14–15 tramos de 2 h supervisados).

## Decisiones para la autora

- **D1. Sensibilidades 1 y 2:** (a) dos corridas, como en el preregistro (≈ 8 h); (b) una
  corrida sin la unión (22 P) más la evaluación por separado de cada bandera sobre las mismas
  predicciones (≈ 4 h); (c) solo evaluación (quitar las 22 P del test de F3.3; minutos, pero
  no responde si el modelo cambia al entrenar sin ellas).
- **D2. Sensibilidad 11:** reentrenar sin las 289 U (responde si el modelo solo aprende el
  balance de carga) o solo evaluar P(ŷ=1|U) sin ellas con las predicciones de F3.3.
- **D3. Esquemas:** los 7 (como F3.3) o `random` + los 4 confirmatorios (ahorra ~6 %; pierde
  `group_system` y `temporal`, que no son confirmatorios).
- **D4. Orden:** propuesta, de más a menos informativa para H1–H2: 8 (recall), 1+2, 11, 13,
  5a, 14, 12, 6, 4. Las B se hacen ya.
- **D5. ¿Antes o después de congelar el preregistro (F4.0)?** El §10 ya fija la lista. Correr
  las sensibilidades antes de F4.0 no cambia el plan confirmatorio, pero cualquier precisión
  de cómo se ejecutan (D1–D3) debe quedar en la bitácora antes de ver sus resultados.

## Cambios de código necesarios (rama aparte, con tests)

- Opción del YAML para excluir composiciones por bandera (en train y test) y para
  reetiquetar (ambiguas como P; ángulo 140°), aplicadas antes de particionar los folds
  internos; las particiones externas de sensibilidad (6, 12) en `data/splits/sensitivity/`,
  sin tocar las congeladas ni sus hashes.
- Varios recalls en una pasada del bucle interno (8).
- Las familias `pre_dft_tau` y `post_dft_r2scan` ya existen en `configs/data.yaml` y `features.sets` (F2.1–F2.3): solo falta el YAML del experimento.
