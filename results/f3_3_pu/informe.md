# Informe F3.3: bagging PU (10 semillas)

> Generado por `scripts/report_f3_3.py` (commit `60fabb5`) a partir de
> `results/f3_3_pu/20261114T164434Z_aacd15c3f6dd` (corrida con el commit `3c24acf`; semillas [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
> última ejecución 87 min) y de las líneas base `results/f2_4_baselines/20261114T013022Z_00a61daabfdf` (mismas semillas).
> No editar a mano. Configuración: `configs/experiments/f3_3_pu.yaml`. Solo `train_pool`.

## 1. Qué se mide

- **Modelo:** bagging PU (Mordelet y Vert, 2014) con 20 bolsas; el aprendiz (RF o GB) y sus
  hiperparámetros se eligen en un bucle interno estratificado de 3 folds entre
  15 candidatos por semilla, por la menor P(ŷ=1) fuera de fold al umbral de recall
  0.90 (preregistro §7.1). El umbral elegido se aplica sin cambios al fold externo.
- **TPR** micro-agregada sobre folds externos; **P(ŷ=1|U)** como control; **TPR a P(ŷ=1)
  fija** por fold con empates repartidos (§7.2). Media de las semillas e IC 95 % (t de
  Student), descriptivos.
- `pu_pre_dft` es el modelo principal; `pu_post_dft` es la cota superior informada por el hull
  (circularidad de la E_hull, regla 2.1.8).

Tamaño de evaluación (esquema `random`, test acumulado sobre folds):

| modelo      | familia   | papel       |   P |   U |
|:------------|:----------|:------------|----:|----:|
| PU post_dft | post_dft  | upper_bound | 193 | 587 |
| PU pre_dft  | pre_dft   | principal   | 193 | 731 |

## 2. TPR en el punto de operación (recall nominal 0,90)

| modelo      | random               | group_system         | loeo_A               | loeo_B               | family               | cluster              | temporal             |
|:------------|:---------------------|:---------------------|:---------------------|:---------------------|:---------------------|:---------------------|:---------------------|
| PU post_dft | 0.892 [0.882, 0.902] | 0.891 [0.877, 0.905] | 0.884 [0.877, 0.892] | 0.876 [0.871, 0.882] | 0.839 [0.817, 0.861] | 0.779 [0.704, 0.853] | 0.811 [0.762, 0.860] |
| PU pre_dft  | 0.895 [0.885, 0.906] | 0.896 [0.882, 0.909] | 0.852 [0.842, 0.862] | 0.847 [0.831, 0.862] | 0.689 [0.657, 0.721] | 0.770 [0.744, 0.796] | 0.744 [0.702, 0.787] |

## 3. P(ŷ = 1 | U) en el punto de operación

| modelo      | random               | group_system         | loeo_A               | loeo_B               | family               | cluster              | temporal             |
|:------------|:---------------------|:---------------------|:---------------------|:---------------------|:---------------------|:---------------------|:---------------------|
| PU post_dft | 0.104 [0.095, 0.112] | 0.107 [0.103, 0.111] | 0.109 [0.104, 0.114] | 0.096 [0.093, 0.100] | 0.123 [0.114, 0.131] | 0.143 [0.122, 0.165] | 0.104 [0.073, 0.135] |
| PU pre_dft  | 0.171 [0.162, 0.180] | 0.174 [0.167, 0.182] | 0.159 [0.155, 0.164] | 0.166 [0.157, 0.175] | 0.150 [0.134, 0.165] | 0.190 [0.170, 0.211] | 0.163 [0.132, 0.194] |

## 4. TPR a P(ŷ = 1) fija

P(ŷ=1) = 0.05:

| modelo      | random               | group_system         | loeo_A               | loeo_B               | family               | cluster              | temporal             |
|:------------|:---------------------|:---------------------|:---------------------|:---------------------|:---------------------|:---------------------|:---------------------|
| PU post_dft | 0.194 [0.191, 0.198] | 0.195 [0.192, 0.197] | 0.137 [0.134, 0.140] | 0.157 [0.152, 0.162] | 0.177 [0.171, 0.182] | 0.167 [0.158, 0.177] | 0.195 [0.173, 0.218] |
| PU pre_dft  | 0.224 [0.220, 0.228] | 0.225 [0.219, 0.231] | 0.141 [0.137, 0.144] | 0.161 [0.157, 0.165] | 0.171 [0.165, 0.177] | 0.184 [0.177, 0.191] | 0.214 [0.190, 0.237] |

P(ŷ=1) = 0.10:

| modelo      | random               | group_system         | loeo_A               | loeo_B               | family               | cluster              | temporal             |
|:------------|:---------------------|:---------------------|:---------------------|:---------------------|:---------------------|:---------------------|:---------------------|
| PU post_dft | 0.380 [0.377, 0.382] | 0.378 [0.374, 0.382] | 0.254 [0.250, 0.259] | 0.274 [0.267, 0.281] | 0.325 [0.317, 0.332] | 0.304 [0.290, 0.317] | 0.383 [0.352, 0.413] |
| PU pre_dft  | 0.428 [0.421, 0.435] | 0.427 [0.416, 0.437] | 0.260 [0.256, 0.265] | 0.289 [0.282, 0.296] | 0.322 [0.314, 0.330] | 0.325 [0.310, 0.341] | 0.381 [0.342, 0.420] |

P(ŷ=1) = 0.20:

| modelo      | random               | group_system         | loeo_A               | loeo_B               | family               | cluster              | temporal             |
|:------------|:---------------------|:---------------------|:---------------------|:---------------------|:---------------------|:---------------------|:---------------------|
| PU post_dft | 0.707 [0.696, 0.717] | 0.710 [0.704, 0.716] | 0.467 [0.464, 0.471] | 0.475 [0.469, 0.480] | 0.570 [0.561, 0.579] | 0.539 [0.516, 0.563] | 0.671 [0.637, 0.705] |
| PU pre_dft  | 0.730 [0.719, 0.740] | 0.735 [0.723, 0.747] | 0.471 [0.465, 0.476] | 0.498 [0.490, 0.505] | 0.541 [0.530, 0.552] | 0.521 [0.497, 0.546] | 0.600 [0.550, 0.650] |


## 5. Brecha de extrapolación (H1, descriptiva)

Estadístico de H1 (§H1): ΔTPR_g = TPR_random − TPR_g en el punto de operación principal
(positivo = caída al extrapolar). Umbral de relevancia preregistrado δ₁ = 0,05. Media de
10 semillas, sin prueba formal (bootstrap jerárquico por grupo en F4):

| modelo      |   TPR random |   ΔTPR loeo_A |   ΔTPR loeo_B |   ΔTPR family |   ΔTPR cluster |
|:------------|-------------:|--------------:|--------------:|--------------:|---------------:|
| PU post_dft |        0.892 |         0.008 |         0.016 |         0.053 |          0.113 |
| PU pre_dft  |        0.895 |         0.044 |         0.049 |         0.206 |          0.125 |

La TPR en el punto de operación mezcla dos efectos: el modelo ordena peor y el umbral
calibrado en el bucle interno deja de dar recall 0,90. A P(ŷ=1) = 0.10 fija solo cuenta la
ordenación, pero la TPR queda acotada por la fracción de positivas de cada fold, que cambia con
el esquema; por eso se da también el máximo alcanzable y el cociente:

| modelo      |                   |   random |   loeo_A |   loeo_B |   family |   cluster |
|:------------|:------------------|---------:|---------:|---------:|---------:|----------:|
| PU post_dft | TPR               |    0.38  |    0.254 |    0.274 |    0.325 |     0.304 |
| PU post_dft | máximo alcanzable |    0.404 |    0.286 |    0.342 |    0.404 |     0.394 |
| PU post_dft | TPR / máximo      |    0.94  |    0.888 |    0.801 |    0.803 |     0.772 |
| PU pre_dft  | TPR               |    0.428 |    0.26  |    0.289 |    0.322 |     0.325 |
| PU pre_dft  | máximo alcanzable |    0.479 |    0.321 |    0.395 |    0.473 |     0.463 |
| PU pre_dft  | TPR / máximo      |    0.894 |    0.81  |    0.73  |    0.681 |     0.702 |

## 5 bis. Frente a τ a igual P(ŷ = 1) (H2, descriptiva)

Estadístico de H2 (§7.2 y §H2): en cada fold, el PU `pre_dft` se umbraliza a la P(ŷ=1) que da
τ < 4,18 sobre las composiciones con τ definido; Δ = TPR_PU − TPR_τ. Margen de no
inferioridad δ₂ = 0,05. Análisis principal (τ *in-sample* en la TableS1, lo que favorece a τ):

| esquema   |   P |   TPR τ |   TPR PU (misma P(ŷ=1)) | Δ (mín.–máx.)            |
|:----------|----:|--------:|------------------------:|:-------------------------|
| random    | 193 |   0.922 |                   0.949 | +0.026 (+0.010 a +0.047) |
| loeo_A    | 193 |   0.922 |                   0.94  | +0.018 (+0.010 a +0.031) |
| loeo_B    | 193 |   0.922 |                   0.917 | -0.005 (-0.016 a +0.005) |
| family    | 193 |   0.922 |                   0.866 | -0.056 (-0.073 a -0.041) |
| cluster   | 193 |   0.922 |                   0.88  | -0.042 (-0.083 a -0.005) |

Sensibilidad sin los compuestos de Bartel et al. (2019) (τ fuera de muestra):

| esquema   |   P |   TPR τ |   TPR PU (misma P(ŷ=1)) | Δ (mín.–máx.)            |
|:----------|----:|--------:|------------------------:|:-------------------------|
| random    |  41 |   0.878 |                   0.912 | +0.034 (+0.000 a +0.098) |
| loeo_A    |  41 |   0.878 |                   0.902 | +0.024 (+0.024 a +0.024) |
| loeo_B    |  41 |   0.878 |                   0.856 | -0.022 (-0.049 a +0.000) |
| family    |  41 |   0.878 |                   0.7   | -0.178 (-0.220 a -0.146) |
| cluster   |  41 |   0.878 |                   0.76  | -0.118 (-0.220 a -0.024) |

## 6. Frente a las líneas base, sobre las mismas composiciones

Cada bloque restringe el modelo PU a las composiciones donde está definida la línea base (su
umbral no cambia). Columnas: TPR en el punto de operación, P(ŷ=1|U) y TPR a P(ŷ=1) = 0.10.
Medias de las semillas [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]; los IC están en `results/f3_3_pu/summary_same_subset.csv`.

### PU pre_dft · todas (train_pool)

Esquemas `random` y confirmatorios:

| modelo                  |   random TPR |   random P(ŷ=1|U) |   random TPR@0.10 |   loeo_A TPR |   loeo_A P(ŷ=1|U) |   loeo_A TPR@0.10 |   loeo_B TPR |   loeo_B P(ŷ=1|U) |   loeo_B TPR@0.10 |   family TPR |   family P(ŷ=1|U) |   family TPR@0.10 |   cluster TPR |   cluster P(ŷ=1|U) |   cluster TPR@0.10 |
|:------------------------|-------------:|------------------:|------------------:|-------------:|------------------:|------------------:|-------------:|------------------:|------------------:|-------------:|------------------:|------------------:|--------------:|-------------------:|-------------------:|
| PU pre_dft              |        0.895 |             0.171 |             0.428 |        0.852 |             0.159 |             0.26  |        0.847 |             0.166 |             0.289 |        0.689 |             0.15  |             0.322 |         0.77  |              0.19  |              0.325 |
| Balance de carga (ICSD) |        0.974 |             0.605 |             0.143 |        0.974 |             0.605 |             0.144 |        0.974 |             0.605 |             0.123 |        0.974 |             0.605 |             0.135 |         0.974 |              0.605 |              0.151 |
| Azar                    |        0.892 |             0.897 |             0.088 |        0.9   |             0.901 |             0.105 |        0.89  |             0.904 |             0.1   |        0.895 |             0.899 |             0.101 |         0.911 |              0.902 |              0.096 |

### PU pre_dft · τ definida

Esquemas `random` y confirmatorios:

| modelo            |   random TPR |   random P(ŷ=1|U) |   random TPR@0.10 |   loeo_A TPR |   loeo_A P(ŷ=1|U) |   loeo_A TPR@0.10 |   loeo_B TPR |   loeo_B P(ŷ=1|U) |   loeo_B TPR@0.10 |   family TPR |   family P(ŷ=1|U) |   family TPR@0.10 |   cluster TPR |   cluster P(ŷ=1|U) |   cluster TPR@0.10 |
|:------------------|-------------:|------------------:|------------------:|-------------:|------------------:|------------------:|-------------:|------------------:|------------------:|-------------:|------------------:|------------------:|--------------:|-------------------:|-------------------:|
| PU pre_dft        |        0.895 |             0.284 |             0.299 |        0.852 |             0.265 |             0.192 |        0.847 |             0.276 |             0.227 |        0.689 |             0.248 |             0.237 |         0.77  |              0.317 |              0.235 |
| τ < 4,18 (Bartel) |        0.922 |             0.415 |             0.181 |        0.922 |             0.415 |             0.114 |        0.922 |             0.415 |             0.166 |        0.922 |             0.415 |             0.149 |         0.922 |              0.415 |              0.154 |

### PU pre_dft · t definida

Esquemas `random` y confirmatorios:

| modelo                       |   random TPR |   random P(ŷ=1|U) |   random TPR@0.10 |   loeo_A TPR |   loeo_A P(ŷ=1|U) |   loeo_A TPR@0.10 |   loeo_B TPR |   loeo_B P(ŷ=1|U) |   loeo_B TPR@0.10 |   family TPR |   family P(ŷ=1|U) |   family TPR@0.10 |   cluster TPR |   cluster P(ŷ=1|U) |   cluster TPR@0.10 |
|:-----------------------------|-------------:|------------------:|------------------:|-------------:|------------------:|------------------:|-------------:|------------------:|------------------:|-------------:|------------------:|------------------:|--------------:|-------------------:|-------------------:|
| PU pre_dft                   |        0.895 |             0.281 |             0.301 |        0.852 |             0.262 |             0.192 |        0.847 |             0.273 |             0.229 |        0.689 |             0.246 |             0.238 |         0.77  |              0.314 |              0.235 |
| t en [0,825; 1,059] (Bartel) |        0.917 |             0.462 |             0.189 |        0.917 |             0.462 |             0.119 |        0.917 |             0.462 |             0.207 |        0.917 |             0.462 |             0.198 |         0.917 |              0.462 |              0.192 |

### PU pre_dft · elegibles post_dft

Esquemas `random` y confirmatorios:

| modelo     |   random TPR |   random P(ŷ=1|U) |   random TPR@0.10 |   loeo_A TPR |   loeo_A P(ŷ=1|U) |   loeo_A TPR@0.10 |   loeo_B TPR |   loeo_B P(ŷ=1|U) |   loeo_B TPR@0.10 |   family TPR |   family P(ŷ=1|U) |   family TPR@0.10 |   cluster TPR |   cluster P(ŷ=1|U) |   cluster TPR@0.10 |
|:-----------|-------------:|------------------:|------------------:|-------------:|------------------:|------------------:|-------------:|------------------:|------------------:|-------------:|------------------:|------------------:|--------------:|-------------------:|-------------------:|
| PU pre_dft |        0.895 |             0.179 |             0.365 |        0.852 |             0.166 |             0.235 |        0.847 |              0.18 |             0.263 |        0.689 |             0.167 |             0.294 |          0.77 |              0.214 |              0.289 |

### PU post_dft · elegibles post_dft

Esquemas `random` y confirmatorios:

| modelo               |   random TPR |   random P(ŷ=1|U) |   random TPR@0.10 |   loeo_A TPR |   loeo_A P(ŷ=1|U) |   loeo_A TPR@0.10 |   loeo_B TPR |   loeo_B P(ŷ=1|U) |   loeo_B TPR@0.10 |   family TPR |   family P(ŷ=1|U) |   family TPR@0.10 |   cluster TPR |   cluster P(ŷ=1|U) |   cluster TPR@0.10 |
|:---------------------|-------------:|------------------:|------------------:|-------------:|------------------:|------------------:|-------------:|------------------:|------------------:|-------------:|------------------:|------------------:|--------------:|-------------------:|-------------------:|
| PU post_dft          |        0.892 |             0.104 |             0.38  |        0.884 |             0.109 |             0.254 |        0.876 |             0.096 |             0.274 |        0.839 |             0.123 |             0.325 |         0.779 |              0.143 |              0.304 |
| E_hull ≤ 0,05        |        0.834 |             0.114 |             0.299 |        0.834 |             0.114 |             0.213 |        0.834 |             0.114 |             0.237 |        0.834 |             0.114 |             0.289 |         0.834 |              0.114 |              0.306 |
| Modelo solo E_hull   |        0.895 |             0.154 |             0.299 |        0.894 |             0.154 |             0.213 |        0.894 |             0.151 |             0.237 |        0.907 |             0.176 |             0.289 |         0.899 |              0.17  |              0.306 |
| Modelo solo simetría |        0.896 |             0.126 |             0.295 |        0.893 |             0.126 |             0.2   |        0.902 |             0.126 |             0.254 |        0.905 |             0.142 |             0.289 |         0.901 |              0.151 |              0.288 |


## 7. Aprendices elegidos en el bucle interno

Fracción de folds externos en que ganó Random Forest (el resto, Gradient Boosting):

| modelo      |   random |   group_system |   loeo_A |   loeo_B |   family |   cluster |   temporal |
|:------------|---------:|---------------:|---------:|---------:|---------:|----------:|-----------:|
| PU post_dft |     0.22 |           0.14 |     0.13 |     0.06 |     0.1  |      0.24 |        0.1 |
| PU pre_dft  |     0.26 |           0.14 |     0.18 |     0.24 |     0.17 |      0.16 |        0.1 |

Configuraciones más elegidas (n.º de folds externos, sumando semillas y esquemas):

| modelo      | configuración                                                                                                            |   folds |
|:------------|:-------------------------------------------------------------------------------------------------------------------------|--------:|
| PU post_dft | gradient_boosting: learning_rate=0.1, max_depth=5, max_features=0.5, min_samples_leaf=5, n_estimators=50, subsample=1.0  |     116 |
| PU post_dft | gradient_boosting: learning_rate=0.1, max_depth=5, max_features=0.5, min_samples_leaf=10, n_estimators=50, subsample=0.8 |      69 |
| PU post_dft | gradient_boosting: learning_rate=0.1, max_depth=5, max_features=0.5, min_samples_leaf=1, n_estimators=50, subsample=0.8  |      66 |
| PU pre_dft  | gradient_boosting: learning_rate=0.1, max_depth=5, max_features=0.5, min_samples_leaf=5, n_estimators=50, subsample=1.0  |     109 |
| PU pre_dft  | gradient_boosting: learning_rate=0.1, max_depth=5, max_features=0.5, min_samples_leaf=1, n_estimators=50, subsample=0.8  |      70 |
| PU pre_dft  | gradient_boosting: learning_rate=0.1, max_depth=5, max_features=0.3, min_samples_leaf=5, n_estimators=50, subsample=0.8  |      59 |

## 8. Coste

- 1050 folds externos (semilla × esquema × fold). Última ejecución: 87 min
  (reanudada: True). Si se reanudó, el tiempo de los tramos anteriores no
  está aquí: se registra en la bitácora.
