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
- Trabajos relacionados: LOCO-CV (Meredig et al., 2018), MatFold (Witman y Schindler,
  2025), SynCoTrain (arXiv 2411.12011), Desai et al. (arXiv 2510.06166), Gu et al. (2022),
  Bartel et al. (2019), Jang et al. (2020), Frey et al. (2020), SynthNN.
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
- **(a) Efecto de la partición (H1):** ΔTPR random → grupos para todos los modelos. — F3.3, F4.1, F4.2
- **(b) Modelo composicional frente a τ (H2):** no inferioridad a P(ŷ=1) fija. — F2.4, F3.3, F4.2
- **(c) Grafos frente a composicional y control por simetría (H3).** — F3.5, F2.4, F4.2
- **(d) Validación de formabilidad** (negativos de Bartel, sintetizadas no perovskita; τ *in-sample*). — F3.4
- **(e) Interpretabilidad:** SHAP y estabilidad entre semillas. — F4.3
- **(f) Análisis de sensibilidad:** alta presión/metaestables, estados ambiguos, π, r2SCAN. — F3.2, F3.6

## 5. Discusión
- Implicaciones para el cribado de materiales y recomendaciones de evaluación.
- Limitaciones: SCAR, ruido de la etiqueta, sesgo de MP hacia fases ordenadas, circularidad
  del hull, procedencia de la estructura.
- Desviaciones del preregistro.
- **Alimentan:** F4.1–F4.4.

## 6. Disponibilidad de datos y código; declaración de uso de IA
- Zenodo (código + datos derivados redistribuibles, DOI); restricciones de licencia de ICSD.
- Declaración de uso de Claude Code según la política de la revista (la IA no es autora).
- **Alimentan:** F6.3.
