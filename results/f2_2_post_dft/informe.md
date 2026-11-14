# Informe F2.2: features `post_dft`

> Generado por `scripts/report_f2_2.py` (commit `d54e187`).
> No editar a mano. Cifras de P y U solo sobre `train_pool` (924 composiciones,
> 193 P); el holdout no se toca.

## 1. Definición

`post_dft` = `pre_dft` + **E_hull** + **ΔHf** (por átomo) de la entrada de MP clasificada como
`perovskite` con **menor E_hull**, con una sola funcional (`GGA_GGA+U`).
Decisión de la autora (2026-11-14): el polimorfo perovskita y no el estado fundamental. La
sensibilidad 4 repite lo mismo con r2SCAN, solo en las composiciones que lo tienen.

**Composiciones no elegibles:** las que no tienen entrada perovskita con termodinámica GGA se
**excluyen** de `post_dft`; no se imputan. Todas las P tienen esa entrada, porque su estructura
experimental está en MP, y muchas U no: imputar daría la regla "sin polimorfo perovskita → U",
derivada de la etiqueta. `perovskite_pu.features.sets` rechaza pedirlas.

## 2. Elegibilidad

| clase                       |   n |   elegibles_GGA |   sin_termo_GGA |   con_GGA_sin_perovskita |   elegibles_r2SCAN |
|:----------------------------|----:|----------------:|----------------:|-------------------------:|-------------------:|
| positivas (train_pool)      | 193 |             193 |               0 |                        0 |                126 |
| no etiquetadas (train_pool) | 731 |             587 |              33 |                      111 |                 47 |
| sintetizadas no perovskita  |  84 |              30 |               0 |                       54 |                  1 |
| ambiguas                    |  25 |               9 |               0 |                       16 |                  1 |

- **`post_dft` sobre `train_pool`: 193 P y 587 U.** Sus comparaciones pareadas con
  `pre_dft` y la línea base de solo E_hull usan este mismo subconjunto.
- U sin ninguna termodinámica GGA: 33; elementos más frecuentes: {'Yb': 33, 'Ba': 1, 'Cs': 1, 'K': 1, 'Li': 1}. Son
  compuestos hipotéticos con Yb que en MP solo tienen r2SCAN (decisión: se excluyen).
- U con GGA pero sin entrada `perovskite`: 111; clases de sus entradas, en n.º de entradas: {'non_perovskite': 94, 'perovskite_borderline': 94, 'hexagonal_polytype': 9}.

## 3. Circularidad: la entrada elegida es la experimental

Fracción de composiciones elegibles cuya perovskita de menor energía es una entrada
experimental: P 97.4% · U 0.0% · sintetizadas no perovskita 0.0%.

En las P, la feature es casi siempre la E_hull de la estructura experimental (relajada, con sus
inclinaciones); en las U es la de un prototipo hipotético. Además, el hull de MP lo definen en
parte las fases de ICSD. Por eso `post_dft` es una **cota superior informada por el hull**
(regla 2.1.8) y se acompaña de la línea base de solo E_hull (F2.4). Se audita en F2.3.

## 4. Polimorfo perovskita frente al estado fundamental (GGA)

ΔE = E_hull(perovskita) − E_hull(estado fundamental), en eV/átomo (mediana [Q1, Q3]).

| clase                       |   elegibles | perovskita_es_fundamental   | ΔE (eV/átomo)        |
|:----------------------------|------------:|:----------------------------|:---------------------|
| positivas (train_pool)      |         193 | 85.0%                       | 0.000 [0.000, 0.000] |
| no etiquetadas (train_pool) |         587 | 91.1%                       | 0.000 [0.000, 0.000] |
| sintetizadas no perovskita  |          30 | 0.0%                        | 0.242 [0.085, 0.426] |
| ambiguas                    |           9 | 22.2%                       | 0.076 [0.011, 0.093] |

## 5. Distribución de E_hull y ΔHf (elegibles; mediana [Q1, Q3])

| clase                       | E_hull (eV/átomo)    | ΔHf (eV/átomo)          |
|:----------------------------|:---------------------|:------------------------|
| positivas (train_pool)      | 0.004 [0.000, 0.036] | -2.876 [-3.292, -2.474] |
| no etiquetadas (train_pool) | 0.334 [0.141, 0.583] | -2.087 [-2.679, -1.361] |
| sintetizadas no perovskita  | 0.269 [0.116, 0.459] | -2.083 [-2.327, -1.590] |
| ambiguas                    | 0.089 [0.072, 0.104] | -1.889 [-2.717, -1.484] |

Es solo descriptiva: la evaluación de la E_hull como clasificador (línea base) se hace en F2.4
con los folds de cada esquema.

## 6. Sensibilidad 4: r2SCAN

Composiciones de `train_pool` con perovskita en GGA y en r2SCAN: 173 (126 P, 47 U).
Solo ellas son elegibles para `post_dft_r2scan`, porque se compara con GGA en el mismo
subconjunto. El análisis es **descriptivo** (decisión de la autora, 2026-11-14) y el subconjunto
está sesgado: además de tener pocas U, estas son casi estables (mediana de su E_hull GGA:
0.000 eV/átomo), porque MP calculó r2SCAN sobre todo para fases estables.

- Spearman de E_hull GGA frente a r2SCAN: 0.495. Es bajo por los empates en 0: sobre
  el hull hay un 70% con GGA y un 83% con r2SCAN. En ΔHf, ρ = 0.986.
- Misma entrada elegida con ambas funcionales: 94.2%.
