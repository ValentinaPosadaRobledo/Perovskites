# Esquema del artículo

**Revista objetivo:** *Digital Discovery* (RSC). Alternativa: *JCIM* (ACS).
**Alcance:** OE1–OE3 (H1–H3). Núcleo: benchmark de sintetizabilidad con control de fuga.
**Título de trabajo:** Evaluación de la generalización de modelos PU learning de
sintetizabilidad de perovskitas óxido ABO₃ bajo extrapolación química.

Cada figura y tabla se genera con un script de `paper/scripts/` a partir de `results/`
(`uv run poe figures`); nada se edita a mano.

---

## 1. Introducción
- Sintetizabilidad frente a estabilidad termodinámica; perovskitas como caso de estudio.
- PU learning en materiales: por qué los no etiquetados no son negativos.
- El problema de la evaluación con particiones aleatorias: sobreestimación al predecir
  química nueva.
- Trabajos relacionados (verificados en `docs/revision_literatura.md`; claves de
  `paper/references.bib`):
  - Particiones fuera de distribución: LOCO-CV (`meredig2018`), MatFold
    (`witman2025matfold`) y "las pruebas OOD suelen interpolar" (`li2025ood`).
  - PU para sintetizabilidad: Gu et al. (`gu2022perovskite`), SynCoTrain
    (`amariamir2025syncotrain`), Desai et al. (`desai2025bridging`), Chung et al.
    (`chung2025solidstate`, el más cercano), Jang et al. (`jang2020structure`,
    `jang2024stoichiometry`), Frey et al. (`frey2019mxenes`), CPUL (`sun2025cpul`) y
    Aydın et al. (`aydin2026tree`).
  - Solo composición y particiones temporales: SynthNN (`antoniuk2023synthnn`) y Zhu et al.
    (`zhu2023synthesizability`).
  - Sesgos y sobrepredicción: `davariashtiyani2024bias`, `schlesinger2026thermodynamic`.
  - Factor de tolerancia: τ (`bartel2019tolerance`).
- Delimitación de la contribución: evaluación sistemática y preregistrada, no un algoritmo
  nuevo.
- **Alimentan:** F0.5.

## 2. Datos
- Fuentes: Materials Project (versión de la BD), Gu et al. (Zenodo), Bartel et al. (2019).
- Identificación de perovskitas por conectividad y validación manual de casos límite.
- Deduplicación estructural y por composición.
- Etiquetado por composición: positivo / sintetizada no perovskita / no etiquetado; banderas.
- Conteos por estructura, elemento A y B y familia.
- **Alimentan:** F1.1–F1.6. **Figuras/tablas:** conteos por familia; tamaños de grupo.

## 3. Métodos
- Features `pre_dft` y `post_dft` y su auditoría (circularidad de E_hull). — F2.1–F2.3
- Líneas base (t, τ, E_hull, solo simetría, trivial). — F2.4
- Bagging PU y prior de clase π. — F3.1, F3.2
- Esquemas de partición y holdout por grupos; demostración de `group_system` ≈ `random`. — F1.7, F1.8
- Protocolo de semillas (`SeedSequence`), CV anidada y punto de operación. — F0.3, F3.3
- Métricas PU (TPR, P(ŷ=1|U), curvas TPR–P(ŷ=1), Lee–Liu solo para ordenar). — F3.3
- Tests estadísticos: Nadeau–Bengio, bootstrap jerárquico, no inferioridad, Holm. — F4.2
- Preregistro (OSF) y reevaluación del modelo de grafos. — F0.4, F4.0, F3.5

## 4. Resultados
> **Estado (2026-11-15):** resultados de validación cruzada sobre `train_pool`, 10 semillas,
> **descriptivos**. Las pruebas formales (bootstrap jerárquico, Holm) son de F4.2, después de
> congelar el preregistro (F4.0); el conjunto reservado (F4.1) sigue sin abrir.

- **(a) Efecto de la partición (H1):** ΔTPR random → grupos para todos los modelos. — F3.3, F4.1, F4.2
  - PU `pre_dft`: TPR 0,895 en `random` → 0,852 `loeo_A`, 0,847 `loeo_B`, **0,689 `family`**,
    **0,770 `cluster`**; caída ≥ δ₁ = 0,05 solo en `family` (0,206) y `cluster` (0,125).
    `post_dft` cae menos (0,053 y 0,113).
  - S1 (§10b): solo `cluster` extrapola de verdad; `loeo_A`, `loeo_B` y `family` son
    "extrapolación débil" (< 50 % de las positivas en grupos extrapolativos), con PCA y sin él.
  - **Confirmatorio (F4.2, bootstrap jerárquico y Holm):** se cumple en `family` (0,206
    [0,139; 0,273]) y `cluster` (0,125 [0,074; 0,173]); no en `loeo_A` (0,044) ni `loeo_B`
    (0,049), con p de Holm 0,082. **H1 no se sostiene en los 4 esquemas** (resultado nulo
    parcial).
  - Sensibilidades (F3.6): `family` y `cluster` superan δ₁ en todas; `loeo_A` queda por
    debajo en las mismas positivas en todas (con las 24 ambiguas como P, 0,061 sobre 217 P).
  - **Figuras:** `paper/figures/fig_tpr_by_scheme.pdf`, `fig_h1_h2_confirmatory.pdf` (a),
    `fig_sensitivity_h1.pdf`.
- **(b) Modelo composicional frente a τ (H2):** no inferioridad a P(ŷ=1) fija. — F2.4, F3.3, F4.2
  - A la P(ŷ=1) de τ: Δ = +0,026 (`random`), +0,018 (`loeo_A`), −0,005 (`loeo_B`),
    **−0,056 (`family`)**, −0,042 (`cluster`). **Resultado nulo probable** para la no
    inferioridad en `family` (la media ya cruza −δ₂). Sin los compuestos de Bartel (τ fuera de
    muestra): −0,178 (`family`) y −0,118 (`cluster`).
  - A P(ŷ=1) fija baja (0,05–0,20), el PU ordena mejor que τ en todos los esquemas (mismas
    composiciones). τ solo iguala o supera al PU en su propio punto de operación, de P(ŷ=1)
    alta (P(ŷ=1|U) ≈ 0,42), y sobre todo cuando se extrapola a familias nuevas.
  - **Confirmatorio (F4.2):** no inferioridad solo en `loeo_A`, con los compuestos de Bartel
    (Δ +0,018, p de Holm 0,0008) y sin ellos (+0,024); no concluyente en `loeo_B`, `family` y
    `cluster`. **No inferioridad no demostrada en los 4 esquemas.**
  - Nadeau–Bengio (no confirmatorio, TPR a P(ŷ=1) = 0,10): en `random` el PU supera a todas
    las líneas base salvo `post_dft`; en `cluster` ninguna diferencia es significativa tras
    Holm.
  - **Figuras:** `fig_h2_delta_tau.pdf`, `fig_tpr_at_fixed_ppr.pdf`,
    `fig_h1_h2_confirmatory.pdf` (b). **Tabla:** `paper/tables/tab_nadeau_bengio.tex`.
- **(c) Grafos frente a composicional y control por simetría (H3).** — F3.5, F2.4, F4.2
  - Línea base de solo simetría ≈ solo E_hull en `random` (TPR 0,896 y 0,895 con
    P(ŷ=1|U) 0,126 y 0,154): la procedencia de la estructura basta para separar P de U.
  - Reproducción de Gu et al.: inferencia exacta con sus pesos (1e-6); S2 en curso.
- **(d) Validación de formabilidad** (negativos de Bartel, sintetizadas no perovskita; τ *in-sample*). — F3.4
  - 73 negativos verdaderos: PU (folds de `family`) FPR 0,308 y AUC 0,750; τ FPR 0,288 y AUC
    0,872, pero *in-sample*: 0,188 en la TableS1 frente a 0,556 fuera. El balance de carga no
    discrimina (AUC ≈ 0,5). Confusiones: politipos hexagonales y cadenas con A grande.
  - **Tabla:** `paper/tables/tab_formability.tex`.
- **(e) Interpretabilidad:** SHAP y estabilidad entre semillas. — F4.3
  - Modelo PU `pre_dft` final (todo `train_pool`; GB en las 10 semillas), TreeSHAP
    *path-dependent* sobre los log-odds de cada bolsa. El indicador de t/μ no definidos (sin
    balance de carga de Bartel: 287 U y ninguna P) reúne el 20 % de la importancia y t el
    18 %; con `compound possible`, ~27 % es información de balance de carga (anterior a la
    síntesis; auditada en F2.3). Estabilidad entre semillas: Spearman 0,95 (grupos) y 0,94
    (features); los 4 primeros, iguales en las 10.
  - **Figura:** `fig_shap_groups.pdf`. **Tabla:** `paper/tables/tab_shap_features.tex`.
- **(f) Análisis de sensibilidad:** alta presión/metaestables, estados ambiguos, π, r2SCAN. — F3.2, F3.6
  - π_U entre 0,09 y 0,18 (Elkan–Noto 0,167; TIcE 0,104), probablemente baja (SCAR no se
    cumple). **Tabla:** `paper/tables/tab_prior.tex`.
  - F3.6 (`results/f3_6_sensibilidad/informe.md`): ninguna sensibilidad cambia el patrón de
    H1. Con τ como feature (14), +0,05 de TPR en `family` y `cluster` fuera de la TableS1 a
    igual P(ŷ=1|U). **Figura:** `fig_sensitivity_h1.pdf`.

## 5. Discusión
- Implicaciones para el cribado de materiales y recomendaciones de evaluación.
- *(Puntos que emergen de F3; a confirmar con F4.)* La partición aleatoria sobreestima sobre
  todo al dejar fuera una familia química entera; los esquemas por elemento apenas
  extrapolan (S1). La ventaja publicada de τ se apoya en parte en datos *in-sample*. La
  estructura (simetría) separa P de U tan bien como la E_hull, lo que obliga a controlar la
  procedencia en cualquier comparación con modelos de grafos.
- Limitaciones: SCAR, ruido de la etiqueta, sesgo de MP hacia fases ordenadas, circularidad
  del hull, procedencia de la estructura.
- Desviaciones del preregistro.
- **Alimentan:** F4.1–F4.4.

## 6. Disponibilidad de datos y código; declaración de uso de IA
- Zenodo (código + datos derivados redistribuibles, DOI); restricciones de licencia de ICSD.
- Declaración de uso de Claude Code según la política de la revista (la IA no es autora).
- **Alimentan:** F6.3.
