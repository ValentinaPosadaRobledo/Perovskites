# Preregistro — BORRADOR

> **Estado:** borrador (F0.4). **No congelado.** Se congela y se deposita en OSF en la
> tarea F4.0, **antes** de abrir el holdout (F4.1). Toda desviación posterior se declara
> explícitamente en el artículo.
>
> **Fijado el 2026-11-13, antes de descargar ningún dato:** δ₁, δ₂, el punto de operación,
> las reglas de decisión de H1–H3 y los parámetros del bootstrap (secciones 7–9).
> Los valores que estaban marcados **[PROPUESTA]** (tolerancias de identificación, k de
> `cluster`, n_iter) quedaron fijados entre el 2026-11-13 y el 2026-11-14, antes de entrenar
> ningún modelo PU (bitácora del ROADMAP).

**Título:** Evaluación de la generalización de modelos de aprendizaje positivo–no
etiquetado (PU learning) para predecir la sintetizabilidad de perovskitas óxido ABO₃ con
control de fuga de información.

**Autora:** Valentina Posada · **Fecha del borrador:** 2026-11-13

---

## 1. Pregunta y contribución

Los modelos de sintetizabilidad se han evaluado sobre todo con particiones aleatorias, que
sobreestiman el rendimiento al predecir química nueva. La contribución **no** es un
algoritmo nuevo: es una evaluación sistemática y preregistrada de la sintetizabilidad PU en
perovskitas óxido ABO₃ bajo extrapolación química, con reevaluación del modelo de grafos
de Gu et al. (2022).

## 2. Definición operativa de la variable de respuesta

- **Tarea:** sintetizabilidad **como perovskita**: dada una composición ABO₃, ¿existe un
  polimorfo con estructura perovskita (octaedros BO₆ que comparten solo vértices) reportado
  experimentalmente?
- **Positivo:** al menos un polimorfo perovskita con registro experimental en ICSD
  (`theoretical == False` en Materials Project, contrastado con `database_IDs["icsd"]`).
- **Sintetizada no perovskita:** en ICSD solo con estructuras no perovskita. Excluida de
  positivos y de no etiquetados; usada solo en la validación de formabilidad.
- **No etiquetado:** sin registro experimental. **No se trata como negativo.**
- **Identificación de perovskitas:** por conectividad (CrystalNN/ChemEnv: B con NC = 6,
  A con NC 8–12, o ≥ 8 O a < 3,3 Å si A tiene par libre; BO₆ que comparten solo vértices;
  ángulo B–O–B ≥ 135°, fijado el 2026-11-13, con 140° como sensibilidad 5),
  contrastada con `StructureMatcher`. Polianiones excluidos; politipos hexagonales 2H/4H/6H
  clasificados aparte. Validación manual de ≥ 50 casos límite.
- **Banderas de sensibilidad:** alta presión/metaestables (MgSiO₃, BiCoO₃, PbVO₃, BiScO₃,
  …) y estados de oxidación ambiguos (BiNiO₃, CaFeO₃, …).
- **Fuente y versión:** Materials Project, versión de la base de datos **2026.09.28**
  (descarga del 2026-11-13, 19:21 UTC; `mp-api` 0.47.2, `emmet-core` 0.88.3, pymatgen
  2026.10.24; registro en `data/MANIFEST.json`); termodinámica `GGA_GGA+U`.

## 3. Alcance y delimitaciones

- Solo óxidos ABO₃ ordenados y estequiométricos presentes en Materials Project.
- Fuera: soluciones sólidas, fases dopadas, de ocupación parcial y deficientes en oxígeno.
- Sin validación experimental propia.
- Las estructuras de ICSD no se redistribuyen; solo identificadores y features derivadas.

## 4. Esquemas de partición

| Esquema | Definición | Papel |
|---|---|---|
| `random` | K-fold aleatorio de composiciones (k = 5) | Referencia para medir la fuga |
| `group_system` | GroupKFold por sistema A–B | Solo diagnóstico (se espera ≈ `random`) |
| `loeo_A` | Leave-one-A-element-out | Extrapolación química |
| `loeo_B` | Leave-one-B-element-out (B con < 5 positivos fusionados por familia) | Extrapolación química |
| `family` | Leave-one-family-out del elemento B (3d, 4d/5d, 4f/5f, grupos principales) | Extrapolación química |
| `cluster` | LOCO-CV, k-means sobre MAGPIE estandarizado, **k = 5** (silueta plana, decisión del 2026-11-14); barrido k = 2–10 descriptivo | Extrapolación química |
| `temporal` | Positivas caracterizadas desde **2009** (año más antiguo de sus referencias de ICSD) solo en test, más el 15 % de U; resto en train | Ampliación |

Se reportan siempre por separado, con la distribución de tamaños de grupo y de positivos
por grupo.

## 5. Conjunto de prueba externo congelado (holdout)

- **Definición:** todas las composiciones cuyo elemento B pertenece a un conjunto reservado
  de elementos, elegidos estratificando por familia hasta cubrir ~10–15 % de los positivos.
- **Semilla:** `holdout_seed = 2026` (independiente de las semillas de experimentos).
- **Elementos B reservados (congelado el 2026-11-14):** Mn (3d); Mo, Pd, Zr (4d/5d); Np, Th
  (4f/5f); Bi, Ge (grupos principales). Holdout: **126 composiciones (33 P, el 14,6 % de las
  positivas; 93 U)**. `train_pool`: 924 composiciones (193 P). Selección en el intento 2.
- **Archivo:** `data/splits/holdout_frozen.csv` (forma canónica: ordenado, LF, UTF-8).
- **SHA-256:** `8eb201620f3851b1e007afb22a1152f584ce344928752d663e65dfc381937e15`
  (coincide con `data/splits/HASHES.json`; creado desde el commit `32535bd`).
- **Acceso:** una sola evaluación (F4.1) mediante `load_holdout()` con `HOLDOUT_UNLOCK=1`,
  después de depositar este preregistro.

## 6. Modelos y líneas base

- **Modelos PU:** bagging PU (Mordelet y Vert, 2014) con Random Forest o Gradient
  Boosting (el aprendiz base se elige en el bucle interno, sección 7.1).
- **Familias de features:** `pre_dft` (MAGPIE completo de Ward et al. (2016), t y μ; **sin
  τ**, decisión del 2026-11-14) y `post_dft` (+ E_hull, ΔHf),
  reportadas por separado; `post_dft` es una cota superior informada por el hull. La E_hull y
  la ΔHf son las del **polimorfo perovskita de menor energía** (GGA_GGA+U), y `post_dft` se
  calcula **solo sobre las composiciones que tienen ese polimorfo** en MP (193 P y 587 U de
  `train_pool`), sin imputar: todas las P lo tienen porque se sintetizaron (2026-11-14). Sus
  comparaciones pareadas con `pre_dft` y la línea base de solo E_hull usan ese subconjunto.
- **Modelo de grafos:** Gu et al. (2022), reevaluado bajo las particiones por grupos.
- **Líneas base** *(definiciones fijadas el 2026-11-14; `configs/experiments/f2_4_baselines.yaml`)*:
  - τ de Bartel et al. (2019): τ < 4,18 (umbral externo; *in-sample* en la TableS1), sobre las
    composiciones con τ definido;
  - t de Goldschmidt: ventana de Bartel 0,825 < t < 1,059 (optimizada en su TableS1, así que
    *in-sample*) como principal y ventana clásica 0,8 < t < 1,0 como sensibilidad, sobre las
    composiciones con t definido;
  - umbral de E_hull (polimorfo perovskita, GGA): E_hull ≤ 0,05 eV/átomo como principal, 0 y
    0,1 como sensibilidad, sobre las composiciones de `post_dft`;
  - modelo de solo E_hull: regresión logística (monótona; equivale a un umbral de E_hull
    calibrado en el bucle interno a recall 0,90);
  - modelo de solo simetría: grupo espacial y sistema cristalino de la misma entrada que
    `post_dft`, regresión logística con codificación one-hot (categorías con < 5 casos en train
    agrupadas) y umbral del bucle interno; es el control de procedencia de H3 y se revisa en
    F3.5 si el modelo de grafos usa otras estructuras;
  - **balance de carga** (añadida el 2026-11-14): estados de oxidación observados en ICSD
    (`oxi_state_guesses` de pymatgen) como principal y estados comunes de pymatgen como
    sensibilidad;
  - clasificador trivial: puntuaciones aleatorias (TPR ≈ P(ŷ = 1)), con el umbral del bucle
    interno.
- **Protocolo** *(fijado el 2026-11-14; `configs/experiments/f3_3_pu.yaml`)*: CV anidada;
  bucle interno con búsqueda aleatoria de presupuesto fijo y selección del umbral; 10
  semillas (0–9).
  - **Candidatos:** n_iter = 15 por semilla, sorteados sin reemplazo de la rejilla conjunta
    de Random Forest (25 árboles por bolsa; max_depth ∈ {sin límite, 5, 10, 20},
    min_samples_leaf ∈ {1, 2, 5, 10}, max_features ∈ {√p, 0,3, 0,5}) y Gradient Boosting
    (50 árboles, tasa 0,1; max_depth ∈ {2, 3, 5}, min_samples_leaf ∈ {1, 5, 10},
    max_features ∈ {√p, 0,3, 0,5}, subsample ∈ {0,8, 1}). Son los mismos para todos los
    esquemas y folds de cada semilla.
  - **Bagging PU:** 20 bolsas; en cada una, todas las P y K = |P| no etiquetadas tomadas
    con reemplazo (flujo `pu_sampling`). Con 20 bolsas, el orden de las puntuaciones
    coincide con el de 100 (Spearman 0,996–0,998); con 50, el coste real (~27 h) duplicaba
    la estimación.
  - **Preprocesamiento** (dentro del Pipeline, ajustado en el train de cada fold): mediana
    con **indicador de dato faltante** (t y μ faltan sin balance de carga de Bartel) y
    eliminación de columnas constantes.

## 7. Punto de operación y métricas *(fijado 2026-11-13)*

### 7.1 Punto de operación principal: recall nominal 0,90 calibrado en el bucle interno

Se usa para H1 y para el reporte de cada modelo PU por separado.

1. En cada fold externo, el bucle interno (3 folds **estratificados aleatorios** sobre el
   train externo, también en los esquemas por grupos; fijado el 2026-11-14) produce
   puntuaciones **fuera de fold** para todas las composiciones del train externo. Así, una
   caída de la TPR externa mide la degradación del umbral al exigir extrapolación.
2. Para cada configuración de hiperparámetros, el umbral t* es el **percentil 10 de las
   puntuaciones fuera de fold de los positivos**: el 90 % de los positivos internos queda
   por encima (`threshold_for_recall` en `perovskite_pu.evaluation.metrics`).
3. **Selección de la configuración:** la de menor P(ŷ = 1) fuera de fold al umbral t*. A
   recall fijo, equivale a maximizar Lee–Liu (que aquí solo **ordena** configuraciones).
   El aprendiz base (Random Forest o Gradient Boosting) es un hiperparámetro más, así que
   "el mejor modelo PU `pre_dft`" queda definido por el propio protocolo y no se elige
   mirando resultados externos.
4. La configuración elegida se reentrena con todo el train externo y t* se aplica **sin
   modificar** al test externo. (El reentrenamiento con más datos puede desplazar algo la
   escala de las puntuaciones; ese efecto forma parte de lo que se mide.)
5. **Métrica:** TPR en el test externo, micro-agregada sobre folds (Σ aciertos / Σ
   positivos), junto a P(ŷ = 1 | U) como control. Si la TPR externa queda por debajo de 0,90,
   eso cuantifica cuánto se degrada un umbral calibrado al exigir extrapolación.

**Justificación.** Un objetivo de recall no depende de la fracción de positivos ni de la
prior π, que todavía se desconocen. El borrador anterior proponía P(ŷ = 1) = 0,10; se
descarta porque, si la fracción de positivos etiquetados en ABO₃ resulta cercana a 0,10, ese
objetivo limitaría la TPR de forma artificial. El valor 0,90 deja margen por debajo de la
TPR = 0,957 reportada por Gu et al. (2022) con partición aleatoria. Recalls nominales de
0,80 y 0,95 se analizan como sensibilidad (sección 10).

### 7.2 Comparaciones entre modelos: a igual P(ŷ = 1) en cada fold de test

Los modelos con umbrales de origen distinto solo se comparan a la misma tasa de predicción
positiva. El umbral de cada modelo se ajusta en el fold de test usando **solo la
distribución de puntuaciones** (no las etiquetas), con `threshold_for_ppr`. Si el corte cae
dentro de un bloque de puntuaciones empatadas (p. ej. una regla binaria), el bloque entra en
la fracción justa: es el valor esperado al romper los empates al azar
(`expected_hits_at_ppr`; precisado el 2026-11-14).

- **H2 (PU frente a τ):** la P(ŷ = 1) de referencia es la que produce τ con su umbral externo
  de 4,18 (τ < 4,18 → perovskita) en ese fold, sobre las composiciones con τ definido. El
  modelo PU se umbraliza para igualarla.
- **H3 (grafos frente a composición):** la P(ŷ = 1) de referencia es la que alcanza el modelo
  PU `pre_dft` en su punto de operación principal (7.1) en ese fold. El modelo de grafos y la
  línea base de solo simetría se umbralizan para igualarla.

### 7.3 Métricas secundarias

- TPR a P(ŷ = 1) fija ∈ {0,05; 0,10; 0,20} y curvas completas TPR frente a P(ŷ = 1) para
  todos los modelos y líneas base.
- Lee y Liu (2003): solo para ordenar modelos (SCAR no se cumple).
- `loeo_A` / `loeo_B`: distribución de la TPR por elemento, además de la micro-agregada.
- Accuracy, F1 y ROC-AUC con U como negativo, etiquetadas como **cotas inferiores**.
- Prior de clase π: ≥ 2 métodos (Elkan–Noto, TIcE o KM2), con análisis de sensibilidad.

## 8. Hipótesis *(márgenes fijados 2026-11-13)*

**Márgenes:** δ₁ = **0,05** y δ₂ = **0,05**, ambos en TPR absoluta. Se usa una única
diferencia mínima relevante de 0,05 en TPR, es decir, 1 de cada 20 perovskitas conocidas.
Si una caída de 0,05 se considera relevante en H1, coherentemente un déficit menor que 0,05
frente a τ se considera irrelevante en H2. Justificación numérica en la sección 8.4.

**Unidad de análisis común.** Todas las diferencias son **pareadas por composición**. Para
cada positivo c del pool de entrenamiento, h_M,e(c) es la fracción de las 10 semillas en
que el modelo M acierta con c en el esquema e. Cada positivo se evalúa exactamente una vez
por semilla en cada esquema. Entonces TPR_M,e = media_c h_M,e(c), que coincide con la TPR
micro-agregada.

**Esquemas confirmatorios:** `loeo_A`, `loeo_B`, `family` y `cluster` (4 esquemas por
hipótesis). `group_system` es diagnóstico y `temporal` es una ampliación; ninguno es
confirmatorio.

### H1 — Efecto de la partición
El TPR en el punto de operación principal cae al pasar de `random` a cada esquema por grupos.

- **Modelo confirmatorio:** el modelo PU `pre_dft` (7.1). Para el modelo de grafos,
  `post_dft` y las líneas base, el mismo contraste se reporta como **descriptivo**.
- **Estadístico:** ΔTPR_g = media_c [h_random(c) − h_g(c)] para cada esquema g.
- **Criterio por esquema:** el valor p bootstrap bilateral, ajustado por Holm sobre los 4
  esquemas, es < 0,05 **y** ΔTPR_g ≥ δ₁ = 0,05.
- **Conclusión global:** H1 se sostiene si el criterio se cumple en los 4 esquemas (test de
  intersección-unión). Si se cumple en algunos, se reporta como apoyo parcial, esquema por
  esquema.
- **Resultado nulo:** publicable; indicaría que la sobreestimación es menor de lo esperado.

### H2 — No inferioridad frente a τ
Bajo los esquemas por grupos, el modelo PU `pre_dft` es no inferior a τ (Bartel et al.,
2019) en TPR a igual P(ŷ = 1) (7.2).

- **Estadístico:** Δ_g = media_c [h_PU,g(c) − h_τ,g(c)], sobre los positivos con τ definido
  (τ es determinista: h_τ ∈ {0, 1}).
- **Criterio por esquema:** la cota inferior del IC 95 % bilateral de Δ_g es > −δ₂ = −0,05
  (equivale a un test unilateral con α = 0,025). La multiplicidad se controla aplicando Holm
  sobre los 4 valores p de H₀: Δ_g ≤ −δ₂.
- **Conclusión global:** no inferioridad en los 4 esquemas (intersección-unión).
- **Superioridad:** solo si además la cota inferior es > 0 (secundaria, también con Holm).
- **Análisis principal:** incluye los compuestos del conjunto de Bartel, donde τ es
  *in-sample*. Eso favorece a τ, así que es conservador para el modelo PU.
- **Sensibilidad:** se repite el análisis excluyendo esos compuestos (τ fuera de muestra).

### H3 — Grafos frente a composición bajo extrapolación
La ventaja del modelo de grafos sobre el modelo PU composicional se reduce al exigir
extrapolación.

- **Estadístico:** D_g = media_c {[h_graf,random(c) − h_comp,random(c)] −
  [h_graf,g(c) − h_comp,g(c)]}, a igual P(ŷ = 1) (7.2).
- **Criterio por esquema:** IC 95 % de D_g > 0 y que excluye 0, con Holm sobre los 4
  esquemas. Conclusión global por intersección-unión, como en H1.
- **Control obligatorio por procedencia de la estructura:** el mismo contraste se calcula
  para la línea base de solo grupo espacial/simetría. **Regla de interpretación
  preregistrada:** si en `random` esa línea base alcanza ≥ 50 % de la ventaja del modelo de
  grafos sobre el composicional, la ventaja del grafo se interpreta como posiblemente debida
  a la procedencia de la estructura (experimental distorsionada frente a prototipo cúbico
  hipotético), y no a la sintetizabilidad.
- **Regla de parada:** si la reproducción del modelo de grafos (F3.5) falla en 3 semanas y el
  plan B (CGCNN/ALIGNN propio) también, H3 se reformula como pregunta exploratoria y se
  declara como desviación.

### 8.4 Justificación de los márgenes: potencia aproximada

Cálculo hecho **antes de tener datos**, con `scripts/potencia_preregistro.py`. Usa una
aproximación normal para la diferencia pareada de aciertos, con efecto de diseño por
agrupamiento químico (~30 elementos B). Supuestos: N positivos, discordancia d (fracción de
positivos en que los dos resultados difieren) y correlación intragrupo ρ ∈ {0; 0,05}.

**H2:** probabilidad de declarar no inferioridad cuando el modelo PU es **exactamente igual**
a τ (N = 250).

| d | δ₂ = 0,03 | δ₂ = 0,05 |
|---|---|---|
| 0,10 | 0,25–0,32 | 0,57–0,71 |
| 0,20 | 0,15–0,18 | 0,33–0,42 |

**H1:** probabilidad de declarar una caída relevante con δ₁ = 0,05 (N = 250, d = 0,20–0,30).

| Caída real | 0,05 | 0,10 | 0,15 |
|---|---|---|---|
| Potencia | 0,24–0,43 | 0,71–0,95 | 0,97–1,00 |

**Conclusiones del cálculo:**

1. Con δ₂ = 0,03 (propuesta inicial), H2 casi siempre terminaría inconclusa aunque el modelo
   PU fuera igual a τ. Por eso se fija δ₂ = 0,05, que además coincide con δ₁.
2. Aun con δ₂ = 0,05, la potencia de H2 es moderada. **Un resultado inconclusivo se reportará
   como tal**, sin reinterpretarlo como inferioridad ni como equivalencia.
3. H1 detecta bien caídas ≥ 0,10. Una caída real cercana a δ₁ tiene, por construcción, ~50 %
   o menos de probabilidad de superar el criterio de estimación puntual.
4. Si en F1.6 el número de positivos resulta muy distinto de 150–350, se recalcula la
   potencia **antes de F4.0**, sin cambiar δ₁ ni δ₂, y se informa en el preregistro.
   *(Comprobado: `train_pool` tiene 193 positivas, dentro de ese rango; no se recalcula.)*

## 9. Análisis estadístico *(fijado 2026-11-13)*

| Uso | Test |
|---|---|
| H1, H2, H3 (confirmatorios, los 4 esquemas por grupos) | Bootstrap jerárquico pareado por composición (abajo) |
| Comparaciones secundarias entre modelos dentro de `random` y `cluster` (CV repetida) | t-test corregido de Nadeau y Bengio |
| Multiplicidad | Holm (α = 0,05) dentro de cada hipótesis, sobre sus 4 esquemas |

**Bootstrap jerárquico:**

- **B = 10 000** remuestreos; IC por percentiles.
- **Nivel 1:** se remuestrean con reemplazo las unidades químicas. Es el elemento A en
  `loeo_A` y el elemento B en `loeo_B`, `family` y `cluster`. En `family`, el remuestreo de
  elementos B se estratifica por familia, porque con solo 4 familias no se pueden
  remuestrear las familias mismas.
- **Nivel 2:** se remuestrean con reemplazo los positivos dentro de cada unidad elegida.
- Por composición se usa h(c), la media sobre las 10 semillas. La variabilidad entre
  semillas se reporta aparte, como DE.
- **Valor p:** bilateral, 2·min{P*(Δ* ≤ 0), P*(Δ* ≥ 0)}. En H2 es unilateral,
  P*(Δ* ≤ −δ₂).
- **Semilla:** `bootstrap_seed: 20261113` en `configs/seeds.yaml`, distinta de todas las
  demás semillas del proyecto.

**Implementación** *(precisada antes de congelar; no cambia las reglas anteriores)*:
`perovskite_pu.evaluation.stats`, con tests sobre datos sintéticos.

- Cada bootstrap (hipótesis × esquema) usa su propio flujo aleatorio, derivado de
  `bootstrap_seed` con `SeedSequence(spawn_key=(hipótesis, esquema))`: es reproducible aunque
  se calcule por separado o en otro orden.
- En el nivel 2, una unidad elegida dos veces en el nivel 1 se remuestrea de forma
  independiente cada vez. El estadístico es la media sobre todas las composiciones
  remuestreadas (ponderación por composición, como la TPR micro-agregada).
- En `loeo_B`, la unidad es el elemento B aunque los grupos pequeños se fusionen por familia
  para formar los folds.
- A igual P(ŷ = 1) (H2, H3), h(c) usa el mismo reparto de empates que §7.2: la probabilidad de
  que cada composición entre en la fracción fijada (`hit_probability_at_ppr`).
- H2: no inferioridad si el valor p unilateral P*(Δ* ≤ −δ₂), ajustado por Holm sobre los 4
  esquemas, es < 0,025; superioridad (secundaria) con P*(Δ* ≤ 0), también con Holm.
- Un valor p bootstrap igual a 0 se informa como < 1/B.

**Comparaciones secundarias con Nadeau–Bengio** *(diseño fijado el 2026-11-20 por la autora,
después de ver F3.3 y antes de calcularlas; no confirmatorias; ver §12)*:

- Modelo de referencia: PU `pre_dft`. Se compara con `post_dft`, el modelo de solo E_hull y
  el de solo simetría (en el subconjunto `post_dft`), τ < 4,18, la ventana de t de Bartel y
  el balance de carga con estados de ICSD.
- Esquemas: `random` y `cluster` (CV repetida con folds aleatorios).
- Métrica por fold: TPR a P(ŷ = 1) = 0,10 (§7.3), con los dos modelos umbralizados sobre las
  composiciones que ambos puntúan y los empates repartidos como en §7.2. Se omiten los folds
  sin positivos en ese subconjunto. J = semillas × folds.
- t corregido con n_test / n_train de las medias del fold de test y del resto en ese
  subconjunto. Holm sobre las 6 comparaciones de cada esquema (α = 0,05, bilateral).

## 10. Análisis de sensibilidad previstos

1. Sin fases de alta presión ni metaestables.
2. Sin casos de estado de oxidación ambiguo.
3. Prior de clase π estimada con métodos distintos.
4. Termodinámica r2SCAN en lugar de GGA/GGA+U (para `post_dft`). **Descriptiva**, solo en las
   composiciones con polimorfo perovskita en ambas funcionales (173 de `train_pool`: 126 P y
   47 U casi estables), por la cobertura sesgada de r2SCAN en MP (2026-11-14).
5. Tolerancias de identificación (ángulo B–O–B, StructureMatcher).
6. Política de grupos pequeños en `loeo_B` (fusionar por familia frente a reportar aparte).
7. Validación de formabilidad (negativos de Bartel y sintetizadas no perovskita).
8. Recall nominal del punto de operación 0,80 y 0,95 (en lugar de 0,90).
9. H2 excluyendo los compuestos del conjunto de Bartel (τ fuera de muestra).
10. S1 en el espacio estandarizado completo, sin PCA.
11. Solo con las U que admiten balance de carga con los estados de oxidación observados en
    ICSD (`flag_no_charge_balance`): las 310 sin balance son no etiquetadas triviales
    (2026-11-14; la definición de los estados se precisó el mismo día).
12. `loeo_A` / `loeo_B` con los sitios asignados por tamaño (r_A > r_B) en lugar de por
    estructura (2026-11-14).
13. Sin las composiciones ambiguas (principal) frente a contarlas como P (2026-11-14).
14. `pre_dft_tau` = `pre_dft` + τ (2026-11-14). τ se ajustó con la TableS1 de Bartel et al.
    (2019), que contiene 152 de las 193 positivas de `train_pool`. Se evalúa en todo el
    conjunto, declarada *in-sample*, y solo sobre las composiciones fuera de la TableS1 (P y U),
    como exige la regla 2.1.6. Fuera de la TableS1 quedan 41 positivas, así que la potencia
    es baja y el análisis es descriptivo.

**Ejecución** *(decidida el 2026-11-15, antes de ver resultados de sensibilidad; F3.6)*: solo
el modelo confirmatorio `pu_pre_dft` (en la 4, `post_dft`), mismo protocolo, semillas y
esquemas que F3.3. Las sensibilidades de exclusión (1, 2, 5, 11) quitan las composiciones de
train y test de los **mismos folds**, con las mismas semillas por tarea, para comparar de
forma pareada por composición con el análisis principal; la 1 y la 2 en corridas separadas, y
la 11 reentrenando. La 5 excluye las 15 positivas que dejan de serlo con 140°; la 6, la 12 y
la 13 usan particiones nuevas generadas con el mismo código y semillas (la 13 sin `temporal`:
las ambiguas no tienen año); la 8 calcula los tres recalls con la misma búsqueda interna. Se
reportan de forma descriptiva (Δ TPR pareada, caída de H1 y Δ de H2), sin pruebas formales.

## 10b. Análisis secundarios (no confirmatorios) *(añadidos 2026-11-13)*

No intervienen en las decisiones sobre H1–H3. Se reportan como contexto y se nombran S1 y S2.

**S1. ¿Los esquemas por grupos extrapolan de verdad?** Motivación: Li et al. (2025) muestran
que muchas pruebas "fuera de distribución" en materiales en realidad interpolan.

- Para cada composición de test, se calcula la distancia euclídea a la composición de train
  más cercana en el espacio MAGPIE, estandarizado y proyectado por PCA al 95 % de la
  varianza. Ambas transformaciones se ajustan solo con el train de cada fold
  (`perovskite_pu.evaluation.extrapolation`).
- Un grupo se considera **extrapolativo** si su mediana de distancia supera el cuantil 0,95
  de las distancias bajo `random`.
- Se reporta, por esquema, la fracción de grupos y de positivos en grupos extrapolativos. Se
  calcula en F1.7, **antes de entrenar**.
- En F4, la correlación de Spearman entre la mediana de distancia del grupo y su TPR, con IC
  bootstrap, es descriptiva.
- **Sensibilidad:** el espacio estandarizado completo, sin PCA.
- **Limitación conocida:** al estandarizar, las features irrelevantes pesan igual que las
  relevantes y diluyen la señal. Se documenta en `tests/test_extrapolation.py`.
- **Interpretación preregistrada:** si un esquema tiene < 50 % de sus positivos en grupos
  extrapolativos, sus resultados en H1 se describen como de "extrapolación débil".

**S2. Efecto de la deduplicación por composición en Gu et al. (F3.5).** Gu et al. (2022)
particiona por **cristal**.

- Antes de aplicar las particiones por grupos, se reproduce su partición aleatoria en dos
  variantes: (a) por cristal, como en el original; (b) con los cristales agrupados por
  composición reducida, para que ninguna composición quede en train y test a la vez.
- **ΔTPR(a − b)** estima la parte de la TPR atribuible a la fuga entre polimorfos. Así se
  separa de la caída por extrapolación química que mide H3.
- Se reporta con un IC bootstrap por composición.

**S3. Control de H3 sin procedencia de la estructura: prototipos cúbicos ideales**
*(añadido 2026-11-19, antes de entrenar ningún modelo de grafos sobre `train_pool`; decisión
de la autora)*. Motivación: con la entrada de MP de menor E_hull, nuestras positivas son
estructuras experimentales inclinadas (B–O–B mediano ≈ 149°) y las no etiquetadas, prototipos
cúbicos (180°); un modelo de grafos puede aprender la procedencia y no la sintetizabilidad
(`docs/borradores/f3_5_h3_estructuras.md`).

- **El análisis confirmatorio de H3 no cambia** (criterio A): estructura = la entrada
  perovskita de menor E_hull de cada composición, la misma de `post_dft` y de la línea base de
  solo simetría.
- **S3 (exploratorio):** el mismo modelo de grafos, con los mismos folds, semillas, bolsas,
  bucle interno y punto de operación, sobre un **prototipo cúbico ideal Pm-3m generado para
  todas las composiciones** de `train_pool`: A en (0, 0, 0), B en (½, ½, ½) y O en
  (½, ½, 0), (½, 0, ½) y (0, ½, ½).
- **Parámetro de red:** a = 2 (r_B + r_O), con r_B el radio de Shannon asignado a B en
  `features/tolerance.py` (el mismo que usa t) y r_O = 1,40 Å. Donde t no está definido, a es
  la mediana de los a definidos en `train_pool` (no usa etiquetas); se informa cuántas
  composiciones la usan.
- Con estos prototipos, la estructura no distingue P de U salvo por la composición: S3 mide si
  el grafo aporta algo frente al PU composicional **sin** información de procedencia.
- Se informa el mismo contraste D_g de H3 (§8) con su IC bootstrap, sin decisión ni Holm. No
  sustituye a la regla de interpretación por procedencia de H3; la complementa.
- Las estructuras de solo teóricas (opción C del borrador, 60 de 193 positivas) **no** se
  usan.
- *(Comprobado al generar los prototipos, 2026-11-20, antes de entrenar.)* La mediana
  (a = 4,08 Å) la usan 343 composiciones: 342 U y 1 P, porque t no está definido sin balance
  de carga de Bartel, que falta en 287 U y en ninguna P. El valor de a lleva así la misma
  información que el indicador de t faltante del PU `pre_dft` (derivada de la composición,
  anterior a la síntesis). Se mantiene la regla, por decisión de la autora, para que la
  comparación con `pre_dft` use la misma información.

**Protocolo del modelo de grafos en H3 y S3** *(fijado 2026-11-20, antes de entrenar;
decisiones de la autora)*:

- Arquitectura e hiperparámetros **fijos de Gu et al.**: GCNN de 64 features y 1
  convolución, grafos de su `CIFData` (radio 7 Å), Adam con lr 0,001, lote 512, 50 épocas y
  la época de mejor AUC de validación. Sin búsqueda de hiperparámetros.
- **Desde cero:** sin sus pesos base preentrenados con datos de MP, para que el grafo no
  traiga información de DFT a una comparación con el PU `pre_dft`.
- **Bagging PU con 20 bolsas**, como el PU composicional: todas las P del train externo y
  K = |P| no etiquetadas con reemplazo; el 10 % de las P y de las U de cada bolsa, para
  validación (como su partición 80/10/10). Puntuación = media de las 20 bolsas.
- **Sin bucle interno:** H3 compara a igual P(ŷ=1) que el PU en cada fold (§7.2); el TPR del
  grafo en su propio punto de operación no se informa.
- Esquemas: `random` y los cuatro confirmatorios, 10 semillas. Con el criterio A, solo las
  composiciones con entrada perovskita en MP (193 P y 587 U); el PU de referencia se toma en
  esas mismas composiciones.

## 11. Decisiones contingentes preregistradas

- **Si hay < ~100 composiciones ABO₃ positivas (F1.6):** se amplía a todas las clases de
  perovskita y los óxidos se evalúan por separado.
- **Si MatFold no es compatible (F1.7):** implementación propia (`src/perovskite_pu/splits/`).

## 12. Registro de desviaciones

| Fecha | Desviación | Motivo | Efecto esperado |
|---|---|---|---|
| 2026-11-14 | **20 bolsas** en el bagging PU en lugar de las 50 propuestas (§6) | Coste: con 4 núcleos en paralelo, 50 bolsas eran ~27 h para las 10 semillas. Decisión por coste, antes de analizar resultados del modelo PU | Ninguno detectable: control 20 frente a 50 en 100 folds externos, ΔTPR +0,001 (`pre_dft`) y +0,018 (`post_dft`), Spearman 0,995–0,996 cuando se elige el mismo candidato |
| *(fecha de F4.0)* | **El preregistro se congela y deposita después de ver los resultados de validación cruzada** de F3.2–F3.4 (y de las sensibilidades de F3.6 que estén hechas), no antes de entrenar | El plan solo exigía congelarlo antes de abrir el conjunto de prueba reservado (F4.1); el depósito en OSF se dejó para el final de F3 | Las reglas confirmatorias (§7–9) están fechadas en el historial de git antes de descargar los datos (commit `e9ff6bc`, 2026-11-13 14:57 UTC; datos a las 19:21) y no cambiaron después. El conjunto reservado sigue sin abrir |
| 2026-11-19 | **Se añade S3** (§10b): control exploratorio de H3 con prototipos cúbicos ideales para todas las composiciones | Con la entrada de menor E_hull, la procedencia (experimental inclinada frente a prototipo cúbico) separa casi por completo P de U; S3 permite separar estructura de procedencia | Ninguno sobre las decisiones confirmatorias: H3 se decide con el criterio A, como estaba previsto. Duplica el cómputo del modelo de grafos. Decidido antes de entrenar ningún grafo sobre `train_pool` |
| 2026-11-20 | **Modelo de grafos sin la transferencia de Gu** (desde cero), sin bucle interno y con 20 bolsas PU en vez de sus 100 modelos | Sus pesos base vienen de datos de MP (información de DFT en una comparación con `pre_dft`); el bucle interno no hace falta para comparar a igual P(ŷ=1); 20 bolsas, como el PU composicional | El grafo puede rendir menos que en su artículo; se reproduce su arquitectura, no su preentrenamiento. Decidido antes de entrenar |
| 2026-11-20 | **Pares y métrica de las comparaciones secundarias de Nadeau–Bengio** (§9) fijados después de ver F3.3 | §9 nombraba el test y los esquemas, pero no los pares ni la métrica | Ninguno sobre H1–H3: son comparaciones no confirmatorias. Se fijaron antes de calcularlas, con las líneas base de §6 salvo el umbral fijo de E_hull (ordena casi igual que el modelo de solo E_hull: Spearman 0,997), el clasificador trivial y las variantes de sensibilidad |
| *(si aplica)* | F3.5 (modelo de grafos) incompleta al congelar | Pendiente de cómputo | Si F3.5 y su plan B fallan en 3 semanas, H3 pasa a exploratoria (regla de parada de §8) |
