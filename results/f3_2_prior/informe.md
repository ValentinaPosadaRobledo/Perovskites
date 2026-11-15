# Informe F3.2: prior de clase π

> Generado por `scripts/report_f3_2.py` (commit `9d2608f`) a partir de
> `results/f3_2_prior/20261115T170049Z_19f79970b85a` (corrida con el commit `60fabb5`; semillas [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]; 103 s).
> No editar a mano. Configuración: `configs/experiments/f3_2_prior.yaml`. Solo `train_pool`.

## 1. Qué se mide

- **c** = P(s = 1 | y = 1), la fracción de perovskitas sintetizables que ya están
  etiquetadas. De ella, bajo SCAR: α = n_L / (c · N) y **π_U** = n_L (1 − c) / (c · n_U), la
  fracción de perovskitas ocultas entre las no etiquetadas.
- **Elkan–Noto:** media de g(x) ≈ P(s = 1 | x) sobre las positivas, cada una puntuada por un
  clasificador calibrado que no la vio (validación cruzada de 5 folds).
- **TIcE** (Bekker y Davis, 2018, Algorithm 1): mayor cota inferior de c en las subregiones
  de un árbol inducido con un fold y evaluado con los demás; valores por defecto del artículo.
- Features `pre_dft`. Media de 10 semillas, IC 95 % (t de Student) y, entre
  paréntesis, mínimo–máximo. La variación entre semillas solo refleja folds y modelos: el
  conjunto de datos es siempre el mismo.

## 2. Resultados

**todo train_pool** (n_L = 193, n_U = 731; c ≥ n_L / N = 0.209 por construcción):

| método                     | papel        | c                                  | π_U                                | positivas ocultas        |   recortes |
|:---------------------------|:-------------|:-----------------------------------|:-----------------------------------|:-------------------------|-----------:|
| Elkan–Noto (RF, isotónica) | principal    | 0.613 [0.608, 0.618] (0.604–0.628) | 0.167 [0.163, 0.170] (0.156–0.173) | 122 [119, 124] (114–127) |          0 |
| Elkan–Noto (GB, isotónica) | sensibilidad | 0.631 [0.625, 0.637] (0.619–0.648) | 0.155 [0.151, 0.159] (0.143–0.163) | 113 [110, 116] (105–119) |          0 |
| Elkan–Noto (RF, sigmoide)  | sensibilidad | 0.591 [0.587, 0.594] (0.584–0.600) | 0.183 [0.181, 0.186] (0.176–0.188) | 134 [132, 136] (129–137) |          0 |
| TIcE (rango en 4 partes)   | principal    | 0.719 [0.703, 0.734] (0.687–0.743) | 0.104 [0.096, 0.111] (0.091–0.120) | 76 [70, 81] (67–88)      |          0 |
| TIcE (cuantiles)           | sensibilidad | 0.749 [0.724, 0.774] (0.685–0.796) | 0.089 [0.077, 0.101] (0.068–0.121) | 65 [57, 74] (49–89)      |          0 |

**sin U sin balance de carga** (sensibilidad 11; n_L = 193, n_U = 442):

| método                     | papel        | c                                  | π_U                                | positivas ocultas        |   recortes |
|:---------------------------|:-------------|:-----------------------------------|:-----------------------------------|:-------------------------|-----------:|
| Elkan–Noto (RF, isotónica) | principal    | 0.603 [0.595, 0.610] (0.589–0.619) | 0.288 [0.280, 0.297] (0.269–0.305) | 127 [124, 131] (119–135) |          0 |
| Elkan–Noto (GB, isotónica) | sensibilidad | 0.634 [0.626, 0.643] (0.622–0.657) | 0.252 [0.243, 0.261] (0.228–0.265) | 111 [107, 115] (101–117) |          0 |
| Elkan–Noto (RF, sigmoide)  | sensibilidad | 0.582 [0.577, 0.588] (0.573–0.596) | 0.314 [0.307, 0.321] (0.296–0.325) | 139 [135, 142] (131–144) |          0 |
| TIcE (rango en 4 partes)   | principal    | 0.780 [0.769, 0.791] (0.757–0.811) | 0.123 [0.116, 0.131] (0.102–0.140) | 54 [51, 58] (45–62)      |          0 |
| TIcE (cuantiles)           | sensibilidad | 0.773 [0.758, 0.788] (0.732–0.813) | 0.129 [0.118, 0.140] (0.100–0.160) | 57 [52, 62] (44–71)      |          0 |

## 3. Diagnósticos

Si el clasificador de Elkan–Noto está calibrado, la media de g en todas las filas debe ser
≈ n_L / N. En TIcE, la primera pasada usa la cota del peor caso (ĉ = 0,5) y la segunda, la
estimación de la primera.

| población         | método                     | media de g (todas)   | n_L / N   | c 1.ª pasada   | DE entre folds   |
|:------------------|:---------------------------|:---------------------|:----------|:---------------|:-----------------|
| all               | Elkan–Noto (RF, isotónica) | 0.208                | 0.209     | —              | —                |
| all               | Elkan–Noto (GB, isotónica) | 0.209                | 0.209     | —              | —                |
| all               | Elkan–Noto (RF, sigmoide)  | 0.208                | 0.209     | —              | —                |
| all               | TIcE (rango en 4 partes)   | —                    | —         | 0.698          | 0.037            |
| all               | TIcE (cuantiles)           | —                    | —         | 0.725          | 0.045            |
| charge_balanced_u | Elkan–Noto (RF, isotónica) | 0.303                | 0.304     | —              | —                |
| charge_balanced_u | Elkan–Noto (GB, isotónica) | 0.304                | 0.304     | —              | —                |
| charge_balanced_u | Elkan–Noto (RF, sigmoide)  | 0.304                | 0.304     | —              | —                |
| charge_balanced_u | TIcE (rango en 4 partes)   | —                    | —         | 0.751          | 0.029            |
| charge_balanced_u | TIcE (cuantiles)           | —                    | —         | 0.744          | 0.048            |

## 4. Sensibilidad: precisión en el punto de operación (bajo SCAR)

Si las perovskitas ocultas se detectaran con la misma TPR que las etiquetadas (SCAR), las
positivas ocultas predichas serían TPR · π_U · n_U. Precisión entre las U predichas =
TPR · π_U / P(ŷ=1|U); precisión global, incluyendo las P del test. Es una aproximación de
sensibilidad, no una métrica del protocolo.

Con π_U entre 0.091 y 0.173 (mín. y máx. de los dos métodos principales en las 10 semillas) y la TPR y P(ŷ=1|U) medias de F3.3 (`results/f3_3_pu/20261114T164434Z_aacd15c3f6dd`, 10 semillas):

| modelo     | esquema      |   TPR |   P(ŷ=1|U) | precisión entre U predichas   | precisión global   |
|:-----------|:-------------|------:|-----------:|:------------------------------|:-------------------|
| PU pre_dft | random       | 0.895 |      0.171 | 0.48–0.91                     | 0.78–0.96          |
| PU pre_dft | group_system | 0.896 |      0.174 | 0.47–0.89                     | 0.77–0.95          |
| PU pre_dft | loeo_A       | 0.852 |      0.159 | 0.49–0.93                     | 0.79–0.97          |
| PU pre_dft | loeo_B       | 0.847 |      0.166 | 0.47–0.89                     | 0.77–0.95          |
| PU pre_dft | family       | 0.689 |      0.15  | 0.42–0.80                     | 0.74–0.91          |
| PU pre_dft | cluster      | 0.77  |      0.19  | 0.37–0.70                     | 0.69–0.86          |
| PU pre_dft | temporal     | 0.744 |      0.163 | 0.42–0.79                     | 0.73–0.90          |

Solo `pu_pre_dft`: π_U se estimó en todo `train_pool`, que es su población de evaluación. `pu_post_dft` se evalúa en las elegibles de post_dft, con otra fracción de ocultas. En `temporal`, el test solo contiene las positivas posteriores al corte, así que la aproximación es más burda.

## 5. Lectura (descriptiva)

- **SCAR no se cumple** y el sesgo tiene dirección: las perovskitas conocidas se concentran
  en las químicas más estudiadas, donde la probabilidad de estar etiquetada es mayor. Ambos
  métodos tienden entonces a **sobrestimar c y subestimar π**: TIcE porque toma la región con
  la mayor fracción de etiquetadas, y Elkan–Noto porque promedia g sobre las etiquetadas, que
  provienen sobre todo de esas regiones. Las π_U de este informe se leen como orientativas y
  probablemente bajas.
- La diferencia entre métodos es parte del resultado: π no está identificada sin supuestos
  fuertes, y por eso el protocolo principal no depende de ella (punto de operación por recall).
