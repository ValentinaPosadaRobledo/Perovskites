# Informe F2.3: auditoría de features

> Generado por `scripts/report_f2_3.py` (commit `1b23e0b`).
> No editar a mano. Especificación: `configs/feature_audit.yaml`. Cifras sobre `train_pool`
> (924 composiciones, 193 P); el holdout no se toca.

## 1. Grupos de features

| grupo                   |   columnas | etapa                | dependencia de la etiqueta   | familias                                        |
|:------------------------|-----------:|:---------------------|:-----------------------------|:------------------------------------------------|
| element_property_magpie |        132 | previa a la síntesis | ninguna                      | pre_dft, pre_dft_tau, post_dft, post_dft_r2scan |
| stoichiometry           |          6 | previa a la síntesis | ninguna                      | pre_dft, pre_dft_tau, post_dft, post_dft_r2scan |
| valence_orbital         |          8 | previa a la síntesis | ninguna                      | pre_dft, pre_dft_tau, post_dft, post_dft_r2scan |
| ion_property            |          3 | previa a la síntesis | ninguna                      | pre_dft, pre_dft_tau, post_dft, post_dft_r2scan |
| geometric_t_mu          |          2 | previa a la síntesis | ninguna                      | pre_dft, pre_dft_tau, post_dft, post_dft_r2scan |
| geometric_tau           |          1 | previa a la síntesis | **in-sample**                | pre_dft_tau                                     |
| thermo_gga              |          2 | DFT (MP)             | **indirecta**                | post_dft                                        |
| thermo_r2scan           |          2 | DFT (MP)             | **indirecta**                | post_dft_r2scan                                 |

- **`element_property_magpie`.** Fuente: Propiedades elementales de Magpie (Ward et al., 2016), vía matminer. Mínimo, máximo, rango, media, desviación media y moda de 22 propiedades elementales, ponderadas por la fracción atómica. Algunas propiedades elementales (GSvolume_pa, GSbandgap, GSmagmom) son DFT de los elementos puros: no dependen de nuestras composiciones.
- **`stoichiometry`.** Fuente: matminer Stoichiometry (normas p de las fracciones atómicas). Constantes en ABO3; las quita el umbral de varianza del Pipeline.
- **`valence_orbital`.** Fuente: matminer ValenceOrbital (electrones s, p, d, f de valencia; tablas de Magpie).
- **`ion_property`.** Fuente: matminer IonProperty: carácter iónico (electronegatividad) y 'compound possible'. 'compound possible' usa los estados de oxidación COMUNES de pymatgen, una lista curada que no se ajusta con nuestras etiquetas. Falla en el 31 % de las P (p. ej. LaMnO3).
- **`geometric_t_mu`.** Fuente: Radios de Shannon (1976) con la asignación de sitios y oxidación de Bartel et al. (2019), solo a partir de la fórmula. NO se usan las columnas A/B del pool, que salen de la estructura (y, en las P, de sus entradas experimentales). NaN si no hay combinación de oxidación que equilibre la carga (39 % de las U y 0 % de las P): se imputa dentro del Pipeline.
- **`geometric_tau`.** Fuente: τ de Bartel et al. (2019): forma funcional (SISSO) y umbral 4,18 ajustados con su TableS1. *Mitigación:* Fuera de la familia principal. Solo en pre_dft_tau (sensibilidad 14), evaluada en todo el conjunto (declarada in-sample) y sin la TableS1 (152 de las 193 P de train_pool están en ella). Como línea base (H2), el análisis principal es conservador para el modelo PU.
- **`thermo_gga`.** Fuente: Endpoint thermo de MP (GGA_GGA+U), polimorfo perovskita de menor E_hull. *Mitigación:* Familia aparte (post_dft), presentada como cota superior informada por el hull, junto a la línea base de solo E_hull. Solo composiciones con polimorfo perovskita calculado (sin imputar). Circularidad: el hull lo definen en parte fases de ICSD y, en el 97 % de las P, la entrada elegida es la estructura experimental. La ΔHf lleva las correcciones MP2020, ajustadas a entalpías experimentales a nivel de población, no por compuesto.
- **`thermo_r2scan`.** Fuente: Endpoint thermo de MP (r2SCAN), mismo criterio de polimorfo. *Mitigación:* Solo sensibilidad 4 (descriptiva), con composiciones que tienen ambas funcionales.

## 2. Comprobaciones automáticas

Toda columna de toda familia pertenece a un grupo auditado (lista blanca), ninguna coincide con
un patrón prohibido, y `pre_dft` solo lleva grupos previos a la síntesis sin dependencia de la
etiqueta. **Resultado: sin problemas.** Las mismas comprobaciones están en
`tests/test_leakage.py`: la de configuración siempre activa y la de datos con `REQUIRE_DATA=1`.

## 3. Columnas bloqueadas

Existen en los datos, pero nunca pueden ser features (metadatos, procedencia o información
derivada de la etiqueta). `features.sets` solo devuelve las columnas de cada familia.

- `train_pool`: `A`, `A_size`, `B`, `B_size`, `family`, `flag_ambiguous_oxidation`, `flag_bartel_conflict`, `flag_high_pressure`, `flag_icsd_without_exp`, `flag_metastable_dft`, `flag_no_charge_balance`, `flag_only_hexagonal`, `flag_site_rule_disagrees`, `label`, `manual_reviewed`, `metastable_gap_ev`, `n_entries`, `n_experimental`, `n_perovskite_experimental`, `n_unique_structures`, `year`
- `f1_5_entries`: `a_element`, `b_element`, `batch_id`, `cls`, `cls_final`, `crystal_system`, `experimental`, `icsd_ids`, `is_gnome`, `manual_reviewed`, `material_id`, `n_icsd_ids`, `spacegroup_number`, `spacegroup_symbol`, `summary_energy_above_hull`, `theoretical`
- `features_pre_dft_meta`: `bartel2019_exp_label`, `bartel2019_is_train`, `bartel_A`, `bartel_A_cn`, `bartel_B`, `bartel_B_cn`, `bartel_X`, `bartel_nA`, `bartel_nB`, `bartel_notes`, `bartel_rA`, `bartel_rB`, `bartel_rX`, `in_bartel2019`
- `features_post_dft_meta`: `eligible_post_dft`, `eligible_post_dft_r2scan`, `experimental_gga`, `experimental_r2scan`, `ground_state_ehull_gga`, `material_id_gga`, `material_id_r2scan`, `n_perovskite_entries_gga`, `n_perovskite_entries_r2scan`

## 4. Cribado univariante (descriptivo; no selecciona features)

AUC de cada columna para separar P de U, con U como negativo (cota inferior) y sin NaN. Se
marca como sospechosa si AUC ≥ 0.95 o ≤ 0.05. **Sospechosas: 0.**

Diez columnas más separadoras de `pre_dft_tau`:

| feature                       |   auc |   nan_frac_P |   nan_frac_U |
|:------------------------------|------:|-------------:|-------------:|
| geom_tau                      | 0.218 |            0 |        0.399 |
| geom_t                        | 0.698 |            0 |        0.393 |
| compound possible             | 0.671 |            0 |        0     |
| MagpieData avg_dev NValence   | 0.336 |            0 |        0     |
| MagpieData range NValence     | 0.355 |            0 |        0     |
| MagpieData mean NUnfilled     | 0.643 |            0 |        0     |
| MagpieData mean NdUnfilled    | 0.641 |            0 |        0     |
| MagpieData avg_dev NdUnfilled | 0.641 |            0 |        0     |
| MagpieData mean MeltingT      | 0.64  |            0 |        0     |
| MagpieData maximum NdUnfilled | 0.639 |            0 |        0     |

- Mejor columna de `pre_dft` (sin τ): `geom_t`, AUC 0.698.
- τ (in-sample): AUC 0.218; es decir, 0.782 para "τ bajo → P".
- Ausencia de t como separador (indicador de dato faltante): AUC 0.304, es decir,
  0.696 para "sin t → U". Es
  información legítima (sin balance de carga con las reglas de Bartel); queda pendiente para
  F3.1 si el imputador añade el indicador.

## 5. Circularidad de E_hull

- E_hull del polimorfo perovskita (elegibles): AUC 0.075, es decir,
  **0.925 para "E_hull baja → P"**, una sola columna frente a 0.698 de la mejor de `pre_dft`.
- Sobre el hull (E_hull = 0): 47.2% de las P y 5.5% de las U elegibles.
- Entrada elegida experimental: 97.4% de las P y 0.0% de las U.

Mecanismo: el hull de MP lo definen en parte las fases de ICSD, y en las P la E_hull es la de
su estructura experimental relajada, mientras que en las U es la de un prototipo hipotético,
a menudo sin relajar sus inclinaciones. Una E_hull baja dice en buena parte "esta fase está en
ICSD". Por eso `post_dft` se presenta solo como **cota superior informada por el hull**, junto
a la línea base de solo E_hull (F2.4), y nunca como evidencia de sintetizabilidad.
