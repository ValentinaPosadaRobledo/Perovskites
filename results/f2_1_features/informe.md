# Informe F2.1: features `pre_dft`

> Generado por `scripts/report_f2_1.py` (commit `03bf247`).
> No editar a mano. Cifras de P y U solo sobre `train_pool` (924 composiciones,
> 193 P); el holdout no se toca.

## 1. Familias de features

Se calculan una vez por composición (1159 composiciones no polianiónicas), porque son
transformaciones fijas de la fórmula. La imputación, el umbral de varianza y el escalado van
dentro del Pipeline de cada fold.

Columnas por grupo: `element_property_magpie` 132, `stoichiometry` 6, `valence_orbital` 8, `ion_property` 3, `geometric` 3.

| familia     |   columnas |   constantes en train_pool |   útiles |
|:------------|-----------:|---------------------------:|---------:|
| pre_dft     |        151 |                         39 |      112 |
| pre_dft_tau |        152 |                         39 |      113 |

- **`pre_dft` (principal):** MAGPIE completo (Ward et al., 2016) + t + μ, **sin τ** (decisión de
  la autora, 2026-11-14).
- **`pre_dft_tau` (sensibilidad 14):** `pre_dft` + τ. Se evalúa en todo el conjunto (declarada
  *in-sample*) y solo sobre las composiciones fuera de la TableS1 (regla 2.1.6).
- Las 39 columnas constantes en `train_pool` son las de `Stoichiometry` (todas las
  composiciones son ABO₃) y varias estadísticas `mode`/`maximum` de MAGPIE que siempre toma
  el O. No aportan información; las elimina el umbral de varianza del Pipeline.

## 2. t, μ y τ: algoritmo y cobertura

Reimplementación del algoritmo de Bartel et al. (2019) (`features/tolerance.py`), validada al
100 % contra los 576 compuestos de su TableS1: mismos A, B, estados de oxidación y radios, y
|Δt| y |Δτ| ≤ 0,005, que es el redondeo de la tabla (`tests/test_tolerance.py`). Los sitios
y la oxidación salen **solo de la fórmula**. Las columnas A/B del pool no se usan, porque salen
de la estructura y en las positivas dependen de la etiqueta.

| clase                       |   n | t_mu_definidos   | tau_definido   | r_A_sustituto   | r_B_sustituto   | en_TableS1   |
|:----------------------------|----:|:-----------------|:---------------|:----------------|:----------------|:-------------|
| positivas (train_pool)      | 193 | 100.0%           | 100.0%         | 48.7%           | 0.0%            | 78.8%        |
| no etiquetadas (train_pool) | 731 | 60.7%            | 60.1%          | 44.6%           | 0.7%            | 5.5%         |
| sintetizadas no perovskita  |  84 | 91.7%            | 91.7%          | 52.4%           | 0.0%            | 72.6%        |
| ambiguas                    |  25 | 100.0%           | 100.0%         | 64.0%           | 0.0%            | 48.0%        |

**Motivos por los que t, μ o τ no están definidos** (quedan NaN y se imputan en el Pipeline):

- **positivas (train_pool):** ninguno
- **no etiquetadas (train_pool):** sin combinación de oxidación que equilibre la carga (287); τ no definido: radio de Shannon no positivo (4); τ no definido: r_A ≤ r_B (1)
- **sintetizadas no perovskita:** sin combinación de oxidación que equilibre la carga (7)
- **ambiguas:** ninguno

La ausencia de t y μ está **muy asociada a la etiqueta**: falta en el
39.3% de las U y en el 0.0% de las P. Equivale a "sin combinación de
oxidación de Bartel que equilibre la carga". No es fuga, porque es una propiedad de la fórmula
anterior a la síntesis, pero sí la misma información que la línea base de balance de carga
(F2.4). Si el Pipeline añade un indicador de dato faltante, el modelo la usa directamente.

## 3. Radios sustitutos

Cuando el ion no tiene radio de Shannon con NC 12 (sitio A) o 6 (sitio B), se usa el NC
disponible más cercano, como hace Bartel. r_A sustituto: 48.7% de las P y
44.6% de las U de `train_pool`. Casos más frecuentes:

| A   |   oxidación |   NC usado |   composiciones |
|:----|------------:|-----------:|----------------:|
| Eu  |           3 |          9 |              22 |
| Dy  |           3 |          9 |              18 |
| Gd  |           3 |          9 |              17 |
| Ho  |           3 |         10 |              17 |
| Er  |           3 |          9 |              16 |
| Tb  |           3 |          9 |              16 |
| Pr  |           3 |          9 |              16 |
| Bi  |           3 |          8 |              15 |
| Y   |           3 |          9 |              14 |
| Mg  |           2 |          8 |              12 |
| Tm  |           3 |          9 |              12 |
| Tl  |           3 |          8 |              12 |
| Yb  |           3 |          9 |              11 |
| Sn  |           2 |          6 |              10 |
| Zn  |           2 |          8 |              10 |

Son sobre todo lantánidos 3+ (NC 9–10), Bi³⁺ y Tl³⁺ (NC 8), Mg²⁺ y Zn²⁺ (NC 8) y Sn²⁺ (NC 6).
El radio sustituto subestima r_A, y con él t y τ, de forma sistemática para esos elementos.
Es un sesgo por elemento, no por etiqueta: la misma composición recibe el mismo radio sea P o
U, y la frecuencia es parecida en ambas clases.

## 4. Sitios de Bartel frente a los de la estructura

En las positivas, el A de Bartel coincide con el de la estructura en el
99.5% de los casos. Discrepancias: TiMnO3 (estructura: A = Mn; Bartel: A = Ti).

## 5. Solapamiento con la TableS1 de Bartel (τ *in-sample*)

- Positivas de `train_pool` en la TableS1: **152**; fuera:
  **41**. Esas 41 positivas son el
  único conjunto donde τ está fuera de muestra (sensibilidad 14 y sensibilidad 9 de H2). Con
  tan pocas, la potencia es baja.
- No etiquetadas en la TableS1: 40 (por `exp_label` de Bartel:
  {'1': 34, '-1': 6}). Las de `exp_label = 1` son las perovskitas
  experimentales de Bartel que MP 2026 solo tiene como teóricas (bitácora, 2026-11-13). Para τ
  también son *in-sample*: la evaluación "fuera de Bartel" excluye toda la TableS1, tanto las
  P como las U.
- Positivas nuestras que Bartel da como NO perovskita: 2 (`flag_bartel_conflict`).

## 6. Dos definiciones de balance de carga

| | P que fallan | U que fallan |
|---|---|---|
| `compound possible` de matminer (estados **comunes** de pymatgen) | 60 de 193 | 477 de 731 |
| `flag_no_charge_balance` de F1.5 (`oxi_state_guesses`: estados **observados en ICSD**) | 5 de 193 | 289 de 731 |

Con los estados comunes fallan perovskitas clásicas como LaMnO₃, porque Mn³⁺ no es "común"
en pymatgen. `compound possible` entra en `pre_dft` como una feature más. La definición de la
línea base de balance de carga se decide en F2.4.

## 7. Distribución descriptiva de t, μ y τ (mediana [Q1, Q3])

| clase                       | geom_t               | geom_mu              | geom_tau             |
|:----------------------------|:---------------------|:---------------------|:---------------------|
| positivas (train_pool)      | 0.907 [0.866, 0.965] | 0.457 [0.432, 0.479] | 2.961 [2.255, 3.862] |
| no etiquetadas (train_pool) | 0.841 [0.786, 0.915] | 0.459 [0.410, 0.543] | 4.407 [3.636, 7.010] |
| sintetizadas no perovskita  | 0.882 [0.802, 1.054] | 0.414 [0.379, 0.457] | 4.790 [4.081, 5.567] |
| ambiguas                    | 0.831 [0.804, 0.909] | 0.461 [0.432, 0.532] | 4.419 [3.689, 5.611] |

Es solo descriptiva, no una evaluación: la de τ como clasificador (línea base) se hace en F2.4
con los folds de cada esquema.
