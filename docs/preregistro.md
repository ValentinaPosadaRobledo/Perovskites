# Preregistro — BORRADOR

> **Estado:** borrador (F0.4, pendiente). **No congelado.** Se congela y se deposita en
> OSF en la tarea F4.0, **antes** de abrir el holdout (F4.1). Toda desviación posterior se
> declara explícitamente en el artículo.
>
> Los valores marcados **[PROPUESTA]** son propuestas iniciales que la autora debe
> confirmar o modificar antes de congelar.

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
| `cluster` | LOCO-CV, k-means sobre MAGPIE (k = 10 **[PROPUESTA]**) | Extrapolación química |
| `temporal` | Entrenar antes de un año de corte, probar después (si hay años de ICSD) | Ampliación |

Se reportan siempre por separado, con la distribución de tamaños de grupo y de positivos
por grupo.

## 5. Conjunto de prueba externo congelado (holdout)

- **Definición:** todas las composiciones cuyo elemento B pertenece a un conjunto reservado
  de elementos, elegidos estratificando por familia hasta cubrir ~10–15 % de los positivos.
- **Semilla:** `holdout_seed = 2026` (independiente de las semillas de experimentos).
- **Elementos B reservados:** *[se fijan en F1.8]*
- **Archivo:** `data/splits/holdout_frozen.csv` (forma canónica: ordenado, LF, UTF-8).
- **SHA-256:** *[se registra en F1.8; debe coincidir con `data/splits/HASHES.json`]*
- **Acceso:** una sola evaluación (F4.1) mediante `load_holdout()` con `HOLDOUT_UNLOCK=1`,
  después de depositar este preregistro.

## 6. Modelos y líneas base

- **Modelos PU:** bagging PU (Mordelet y Vert, 2014) con Random Forest y Gradient Boosting.
- **Familias de features:** `pre_dft` (MAGPIE, t, μ, τ) y `post_dft` (+ E_hull, ΔHf),
  reportadas por separado; `post_dft` es una cota superior informada por el hull.
- **Modelo de grafos:** Gu et al. (2022), reevaluado bajo las particiones por grupos.
- **Líneas base:** t de Goldschmidt, τ de Bartel et al. (2019), umbral de E_hull, modelo de
  solo E_hull, modelo de solo grupo espacial/simetría y clasificador trivial.
- **Protocolo:** CV anidada; bucle interno con búsqueda aleatoria de presupuesto fijo
  (n_iter = 15 **[PROPUESTA]**) y selección del umbral; 10 semillas (0–9).

## 7. Punto de operación y métricas

- **Punto de operación principal:** el umbral elegido en el bucle interno, que apunta a
  P(ŷ = 1) = 0,10 **[PROPUESTA]**.
- **Punto de operación secundario:** TPR a P(ŷ = 1) fija ∈ {0,05; 0,10; 0,20}
  **[PROPUESTA]**, para comparar con τ (umbral externo de 4,18).
- **Métrica principal:** TPR sobre positivos reservados. **Control:** P(ŷ = 1 | U).
- **Lee y Liu (2003):** solo para ordenar modelos (SCAR no se cumple).
- **`loeo_A` / `loeo_B`:** TPR micro-agregado sobre folds + distribución por elemento.
- **Cotas inferiores:** accuracy, F1 y ROC-AUC con U como negativo, etiquetadas como tales.
- **Prior de clase π:** ≥ 2 métodos (Elkan–Noto, TIcE o KM2), con análisis de sensibilidad.

## 8. Hipótesis

### H1 — Efecto de la partición
El TPR en el punto de operación fijo cae al pasar de `random` a cada esquema por grupos
(`loeo_A`, `loeo_B`, `family`, `cluster`).

- **Criterio de apoyo:** el IC 95 % de ΔTPR = TPR_random − TPR_grupo (bootstrap jerárquico
  por grupo) excluye 0 **y** la estimación puntual es ≥ **δ₁ = 0,05 [PROPUESTA]**.
- **Se evalúa para:** modelos PU composicionales, modelo de grafos y líneas base.
- **Resultado nulo:** publicable; indicaría que la sobreestimación es menor de lo esperado.

### H2 — No inferioridad frente a τ
Bajo los esquemas por grupos, el mejor modelo PU `pre_dft` es no inferior a τ (Bartel et
al., 2019) en TPR a igual P(ŷ = 1).

- **Criterio:** test unilateral de no inferioridad; la cota inferior del IC 95 % de
  (TPR_PU − TPR_τ) es > −**δ₂**, con **δ₂ = 0,03 [PROPUESTA]**.
- **Superioridad:** solo si la cota inferior es > 0.
- τ sobre el subconjunto de Bartel se declara **in-sample**.

### H3 — Grafos frente a composición bajo extrapolación
La ventaja del modelo de grafos sobre el mejor modelo composicional se reduce al exigir
extrapolación.

- **Criterio:** [ΔTPR(grafo − composicional)]_random − [ΔTPR(grafo − composicional)]_grupos
  > 0, con IC 95 % que excluya 0.
- **Control obligatorio:** línea base de solo grupo espacial/simetría (riesgo de que el
  grafo aprenda la procedencia de la estructura: experimentales distorsionadas frente a
  prototipos cúbicos hipotéticos).
- **Regla de parada:** si la reproducción del modelo de grafos (F3.5) falla en 3 semanas y
  el plan B (CGCNN/ALIGNN propio) también, H3 se reformula como pregunta exploratoria y se
  declara como desviación.

### OE4 (exploratorio, solo tesis)
Priorización de candidatos por estabilidad acuosa (ΔG_pbx) en condiciones fisiológicas.
**No es una hipótesis** (un enunciado de existencia no es falsable).

## 9. Análisis estadístico

| Comparación | Test |
|---|---|
| CV repetida con folds aleatorios (`random`, `cluster`) | t-test corregido de Nadeau y Bengio |
| Esquemas leave-one-group-out (`loeo_A`, `loeo_B`, `family`) | Bootstrap jerárquico por grupo (B = 10 000 **[PROPUESTA]**) o permutación por grupo |
| H2 | No inferioridad unilateral con margen δ₂ |
| Comparaciones múltiples | Corrección de Holm (α = 0,05) |

## 10. Análisis de sensibilidad previstos

1. Sin fases de alta presión ni metaestables.
2. Sin casos de estado de oxidación ambiguo.
3. Prior de clase π estimada con métodos distintos.
4. Termodinámica r2SCAN en lugar de GGA/GGA+U (para `post_dft`).
5. Tolerancias de identificación (ángulo B–O–B, StructureMatcher).
6. Política de grupos pequeños en `loeo_B` (fusionar por familia frente a reportar aparte).
7. Validación de formabilidad (negativos de Bartel y sintetizadas no perovskita).

## 11. Decisiones contingentes preregistradas

- **Si hay < ~100 composiciones ABO₃ positivas (F1.6):** se amplía a todas las clases de
  perovskita y los óxidos se evalúan por separado.
- **Si MatFold no es compatible (F1.7):** implementación propia (`src/perovskite_pu/splits/`).

## 12. Registro de desviaciones

| Fecha | Desviación | Motivo | Efecto esperado |
|---|---|---|---|
| | *(ninguna; el preregistro aún no está congelado)* | | |
