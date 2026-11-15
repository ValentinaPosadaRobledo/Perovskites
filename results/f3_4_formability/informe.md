# Informe F3.4: validación de formabilidad

> Generado por `scripts/report_f3_4.py` (commit `1197718`) a partir de
> `results/f3_4_formability/20261115T170454Z_b6b371e72440` (corrida con el commit `d662b65`; semillas [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]; 8 min).
> No editar a mano. Configuración: `configs/experiments/f3_4_formability.yaml`. Solo `train_pool`.

## 1. Qué se mide

- **Formabilidad, no sintetizabilidad:** los negativos son óxidos ABO₃ que se han sintetizado
  y no son perovskita. Un buen modelo de "sintetizable como perovskita" debería puntuarlos
  por debajo de las perovskitas conocidas. Aquí hay negativos verdaderos: la FPR y la ROC-AUC
  entre P y negativos son métricas reales, no cotas inferiores.
- **Variante `final`:** el modelo PU entrenado con todo train_pool con el protocolo de F3.3
  (umbral a recall 0,90 en el CV interno). Sus P y U son de entrenamiento: su TPR de
  referencia es la de F3.3.
- **Variante `family`:** cada negativo lo puntúa el modelo del fold de `family` que reserva
  la familia de su elemento B (extrapolación). TPR y AUC usan las P del test de esos folds.
- **Reglas:** τ, ventanas de t y balance de carga, con sus umbrales externos. **τ es
  *in-sample*** en los negativos de la TableS1 (Bartel et al. la ajustaron con ellos).
- Media de 10 semillas e IC 95 % (t de Student); las reglas no varían entre semillas.

Conjuntos (procedencia: nuestra etiquetación, Bartel o ambas):

|                            |   bartel2019 |   both |   ours |
|:---------------------------|-------------:|-------:|-------:|
| sintetizadas no perovskita |            1 |     47 |     25 |
| polianiones (Bartel)       |           61 |      0 |      0 |
| ambiguas (Bartel)          |            8 |      0 |      0 |
| conflicto: P para nosotros |            2 |      0 |      0 |
| conflicto: U para nosotros |            6 |      0 |      0 |
| B reservado (no se puntúa) |            1 |     11 |      1 |

## 2. Modelo final (todo train_pool)

| modelo     | TPR (P)   | P(ŷ=1|U)   | FPR sint. no perov.   | FPR polianiones      | FPR ambiguas         | AUC P vs sint. no perov.   |
|:-----------|:----------|:-----------|:----------------------|:---------------------|:---------------------|:---------------------------|
| PU pre_dft | —         | —          | 0.379 [0.357, 0.402]  | 0.218 [0.185, 0.251] | 0.200 [0.154, 0.246] | —                          |

## 3. Extrapolación: folds de `family` y reglas

| modelo     | TPR (P)              | P(ŷ=1|U)             | FPR sint. no perov.   | FPR polianiones      |   FPR ambiguas | AUC P vs sint. no perov.   |
|:-----------|:---------------------|:---------------------|:----------------------|:---------------------|---------------:|:---------------------------|
| PU pre_dft | 0.689 [0.657, 0.721] | 0.150 [0.134, 0.165] | 0.308 [0.296, 0.321]  | 0.090 [0.069, 0.111] |              0 | 0.750 [0.740, 0.761]       |

Reglas (en todas las P y U de train_pool y en todos los negativos donde están definidas):

| modelo                       |   TPR (P) |   P(ŷ=1|U) |   FPR sint. no perov. |   FPR polianiones |   FPR ambiguas |   AUC P vs sint. no perov. |
|:-----------------------------|----------:|-----------:|----------------------:|------------------:|---------------:|---------------------------:|
| τ < 4,18 (Bartel)            |     0.922 |      0.415 |                 0.288 |             0     |          0.125 |                      0.872 |
| t en [0,825; 1,059] (Bartel) |     0.917 |      0.462 |                 0.394 |             0.377 |          0.125 |                      0.782 |
| t en [0,8; 1,0] (clásica)    |     0.86  |      0.536 |                 0.439 |             0.262 |          0.625 |                      0.762 |
| Balance de carga (ICSD)      |     0.974 |      0.605 |                 0.944 |             0.934 |          1     |                      0.515 |
| Balance de carga (comunes)   |     0.689 |      0.347 |                 0.836 |             0.967 |          0.75  |                      0.427 |

## 4. τ *in-sample*: negativos de la TableS1 frente al resto (sintetizadas no perovskita)

| modelo                       | FPR en la TableS1 (τ in-sample)   | FPR fuera de la TableS1   |
|:-----------------------------|:----------------------------------|:--------------------------|
| PU pre_dft                   | 0.350 (n=48)                      | 0.436 (n=25)              |
| τ < 4,18 (Bartel)            | 0.188 (n=48)                      | 0.556 (n=18)              |
| t en [0,825; 1,059] (Bartel) | 0.333 (n=48)                      | 0.556 (n=18)              |
| t en [0,8; 1,0] (clásica)    | 0.438 (n=48)                      | 0.444 (n=18)              |
| Balance de carga (ICSD)      | 1.000 (n=47)                      | 0.840 (n=25)              |
| Balance de carga (comunes)   | 0.896 (n=48)                      | 0.720 (n=25)              |

## 5. Por familia de B (variante `family`)

| modelo     | familia de B   |   P (test) |   sint. no perov. |   TPR |   FPR |
|:-----------|:---------------|-----------:|------------------:|------:|------:|
| PU pre_dft | 3d             |        108 |                24 | 0.764 | 0.429 |
| PU pre_dft | 4d_5d          |         37 |                14 | 0.908 | 0.643 |
| PU pre_dft | 4f_5f          |         13 |                 4 | 0.169 | 0.5   |
| PU pre_dft | main_group     |         35 |                31 | 0.42  | 0.039 |

## 6. Sintetizadas no perovskita que el modelo final predice como perovskita

En al menos la mitad de las semillas:

| composition   |   fracción de semillas | τ predice perovskita   | en la TableS1   |
|:--------------|-----------------------:|:-----------------------|:----------------|
| BaCoO3        |                    1   | sí                     | sí              |
| BaRhO3        |                    1   | sí                     | no              |
| BaCrO3        |                    1   | sí                     | sí              |
| BaSiO3        |                    1   | no                     | sí              |
| BaRuO3        |                    1   | sí                     | sí              |
| CsNbO3        |                    1   | sí                     | sí              |
| CeAsO3        |                    1   | sí                     | no              |
| BaTcO3        |                    1   | sí                     | no              |
| RbNbO3        |                    1   | sí                     | no              |
| NaIrO3        |                    1   | no                     | no              |
| LaAsO3        |                    1   | sí                     | no              |
| KVO3          |                    1   | no                     | sí              |
| GaBiO3        |                    1   | sí                     | no              |
| VHgO3         |                    1   | no                     | no              |
| VAgO3         |                    1   | no                     | sí              |
| TlSbO3        |                    1   | sí                     | sí              |
| TlVO3         |                    1   | sí                     | sí              |
| YGaO3         |                    1   | sí                     | no              |
| NaVO3         |                    1   | no                     | sí              |
| TePbO3        |                    1   | no                     | sí              |
| RbVO3         |                    1   | no                     | sí              |
| BaTeO3        |                    0.9 | sí                     | sí              |
| RbTaO3        |                    0.9 | sí                     | no              |
| CsVO3         |                    0.9 | no                     | sí              |
| CaPtO3        |                    0.9 | sí                     | no              |
| NaAsO3        |                    0.7 | no                     | sí              |
| BaNiO3        |                    0.6 | no                     | sí              |
| AgSbO3        |                    0.5 | sí                     | sí              |
| MgTiO3        |                    0.5 | no                     | sí              |

## 7. Coherencia con F3.3

9240 puntuaciones de train_pool comparadas con F3.3 (`results/f3_3_pu/20261114T164434Z_aacd15c3f6dd`): diferencia máxima 0.00e+00; umbrales idénticos.
