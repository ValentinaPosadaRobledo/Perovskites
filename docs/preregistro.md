# Preregistro — BORRADOR

> **Estado:** borrador (F0.4). **No congelado.** Se congela y se deposita en OSF en la
> tarea F4.0, **antes** de abrir el holdout (F4.1). Toda desviación posterior se declara
> explícitamente en el artículo.
>
> **Fijado el 2026-11-13, antes de descargar ningún dato:** δ₁, δ₂, el punto de operación,
> las reglas de decisión de H1–H3 y los parámetros del bootstrap (secciones 7–9).
> Los valores que siguen marcados **[PROPUESTA]** (tolerancias de identificación, k de
> `cluster`, n_iter) se fijan en F1 y deben estar cerrados antes de F4.0.

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
  A con NC 8–12, BO₆ que comparten solo vértices, ángulo B–O–B > 140° **[PROPUESTA]**),
  contrastada con `StructureMatcher`. Polianiones excluidos; politipos hexagonales 2H/4H/6H
  clasificados aparte. Validación manual de ≥ 50 casos límite.
- **Banderas de sensibilidad:** alta presión/metaestables (MgSiO₃, BiCoO₃, PbVO₃, BiScO₃,
  …) y estados de oxidación ambiguos (BiNiO₃, CaFeO₃, …).
- **Fuente y versión:** Materials Project, versión de la base de datos: *[se registra en
  F1.1]*; termodinámica `GGA_GGA+U`.

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
- **Líneas base:** t de Goldschmidt, τ de Bartel et al. (2019), umbral de E_hull, modelo de
  solo E_hull, modelo de solo grupo espacial/simetría, **balance de carga** (estados de
  oxidación comunes, como en SynthNN; añadida el 2026-11-14) y clasificador trivial.
- **Protocolo:** CV anidada; bucle interno con búsqueda aleatoria de presupuesto fijo
  (n_iter = 15 **[PROPUESTA]**) y selección del umbral; 10 semillas (0–9).

## 7. Punto de operación y métricas *(fijado 2026-11-13)*

### 7.1 Punto de operación principal: recall nominal 0,90 calibrado en el bucle interno

Se usa para H1 y para el reporte de cada modelo PU por separado.

1. En cada fold externo, el bucle interno (3 folds sobre el train externo) produce
   puntuaciones **fuera de fold** para todas las composiciones del train externo.
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
distribución de puntuaciones** (no las etiquetas), con `threshold_for_ppr`.

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

## 11. Decisiones contingentes preregistradas

- **Si hay < ~100 composiciones ABO₃ positivas (F1.6):** se amplía a todas las clases de
  perovskita y los óxidos se evalúan por separado.
- **Si MatFold no es compatible (F1.7):** implementación propia (`src/perovskite_pu/splits/`).

## 12. Registro de desviaciones

| Fecha | Desviación | Motivo | Efecto esperado |
|---|---|---|---|
| | *(ninguna; el preregistro aún no está congelado)* | | |
