# Informe F2.4: líneas base

> Generado por `scripts/report_f2_4.py` (commit `836d147`, árbol sucio) a partir de
> `results/f2_4_baselines/20261114T013022Z_00a61daabfdf` (corrida con el commit `922443b`; 10 semillas; 20 s).
> No editar a mano. Configuración: `configs/experiments/f2_4_baselines.yaml`. Solo `train_pool`.

## 1. Qué se mide

- **Punto de operación:** reglas con su umbral externo; modelos (solo E_hull, solo simetría,
  azar) con el umbral que da recall 0,90 en un bucle interno estratificado (preregistro §7.1).
- **TPR** micro-agregada sobre folds y **P(ŷ=1|U)** como control; media de 10 semillas e
  IC 95 % (t de Student), descriptivos. Un solo número sin intervalo significa que no varía
  entre semillas.
- Las reglas no se entrenan: cada composición recibe la misma predicción en cualquier
  esquema. Su TPR solo cambia en `temporal`, cuyo test contiene únicamente las positivas
  posteriores al corte.

Tamaño de cada subconjunto de evaluación (esquema `random`, test acumulado sobre folds):

| línea base                   | subconjunto   |   P |   U |
|:-----------------------------|:--------------|----:|----:|
| τ < 4,18 (Bartel)            | tau_defined   | 193 | 439 |
| t en [0,825; 1,059] (Bartel) | t_defined     | 193 | 444 |
| t en [0,8; 1,0] (clásica)    | t_defined     | 193 | 444 |
| E_hull ≤ 0,05                | post_dft      | 193 | 587 |
| E_hull = 0                   | post_dft      | 193 | 587 |
| E_hull ≤ 0,1                 | post_dft      | 193 | 587 |
| Modelo solo E_hull           | post_dft      | 193 | 587 |
| Modelo solo simetría         | post_dft      | 193 | 587 |
| Balance de carga (ICSD)      | all           | 193 | 731 |
| Balance de carga (comunes)   | all           | 193 | 731 |
| Azar                         | all           | 193 | 731 |

## 2. TPR en el punto de operación (líneas base principales)

| línea base                   | random               | group_system         | loeo_A               | loeo_B               | family               | cluster              | temporal             |
|:-----------------------------|:---------------------|:---------------------|:---------------------|:---------------------|:---------------------|:---------------------|:---------------------|
| τ < 4,18 (Bartel)            | 0.922                | 0.922                | 0.922                | 0.922                | 0.922                | 0.922                | 0.852                |
| t en [0,825; 1,059] (Bartel) | 0.917                | 0.917                | 0.917                | 0.917                | 0.917                | 0.917                | 0.815                |
| E_hull ≤ 0,05                | 0.834                | 0.834                | 0.834                | 0.834                | 0.834                | 0.834                | 0.741                |
| Modelo solo E_hull           | 0.895 [0.892, 0.898] | 0.894 [0.890, 0.898] | 0.894 [0.892, 0.896] | 0.894 [0.891, 0.896] | 0.907 [0.905, 0.910] | 0.899 [0.895, 0.903] | 0.774 [0.766, 0.782] |
| Modelo solo simetría         | 0.896 [0.892, 0.900] | 0.901 [0.899, 0.902] | 0.893 [0.889, 0.898] | 0.902                | 0.905 [0.897, 0.913] | 0.901 [0.895, 0.906] | 0.815                |
| Balance de carga (ICSD)      | 0.974                | 0.974                | 0.974                | 0.974                | 0.974                | 0.974                | 0.963                |
| Azar                         | 0.892 [0.875, 0.908] | 0.899 [0.889, 0.910] | 0.900 [0.884, 0.916] | 0.890 [0.872, 0.908] | 0.895 [0.870, 0.920] | 0.911 [0.894, 0.929] | 0.915 [0.875, 0.954] |

## 3. P(ŷ = 1 | U) en el punto de operación (principales)

| línea base                   | random               | group_system         | loeo_A               | loeo_B               | family               | cluster              | temporal             |
|:-----------------------------|:---------------------|:---------------------|:---------------------|:---------------------|:---------------------|:---------------------|:---------------------|
| τ < 4,18 (Bartel)            | 0.415                | 0.415                | 0.415                | 0.415                | 0.415                | 0.415                | 0.400 [0.362, 0.437] |
| t en [0,825; 1,059] (Bartel) | 0.462                | 0.462                | 0.462                | 0.462                | 0.462                | 0.462                | 0.449 [0.406, 0.492] |
| E_hull ≤ 0,05                | 0.114                | 0.114                | 0.114                | 0.114                | 0.114                | 0.114                | 0.118 [0.093, 0.144] |
| Modelo solo E_hull           | 0.154 [0.151, 0.157] | 0.153 [0.152, 0.154] | 0.154 [0.153, 0.154] | 0.151 [0.150, 0.152] | 0.176 [0.175, 0.177] | 0.170 [0.167, 0.173] | 0.152 [0.127, 0.178] |
| Modelo solo simetría         | 0.126 [0.126, 0.126] | 0.144 [0.104, 0.184] | 0.126                | 0.126                | 0.142 [0.106, 0.179] | 0.151 [0.094, 0.208] | 0.141 [0.129, 0.154] |
| Balance de carga (ICSD)      | 0.605                | 0.605                | 0.605                | 0.605                | 0.605                | 0.605                | 0.625 [0.601, 0.650] |
| Azar                         | 0.897 [0.887, 0.907] | 0.905 [0.895, 0.915] | 0.901 [0.889, 0.913] | 0.904 [0.899, 0.908] | 0.899 [0.885, 0.912] | 0.902 [0.891, 0.914] | 0.899 [0.875, 0.923] |

## 4. TPR a P(ŷ = 1) fija = 0.10

Por fold, a P(ŷ=1) exacta según la distribución de puntuaciones del test (sin etiquetas); si
el corte cae en un bloque de puntuaciones empatadas (balance de carga binario, E_hull = 0), el
bloque entra en la fracción justa (valor esperado al romper los empates al azar). TPR máxima
alcanzable si todas las predicciones positivas fueran P: tau_defined 0.327; t_defined 0.330; post_dft 0.404; all 0.479.

| línea base                   | random               | group_system         | loeo_A               | loeo_B               | family               | cluster              | temporal             |
|:-----------------------------|:---------------------|:---------------------|:---------------------|:---------------------|:---------------------|:---------------------|:---------------------|
| τ < 4,18 (Bartel)            | 0.181 [0.175, 0.188] | 0.182 [0.178, 0.185] | 0.114                | 0.166                | 0.149                | 0.154 [0.141, 0.166] | 0.089 [0.062, 0.117] |
| t en [0,825; 1,059] (Bartel) | 0.189 [0.183, 0.195] | 0.189 [0.185, 0.194] | 0.119                | 0.207                | 0.198                | 0.192 [0.188, 0.197] | 0.126 [0.103, 0.149] |
| E_hull ≤ 0,05                | 0.299 [0.296, 0.302] | 0.299 [0.297, 0.300] | 0.213                | 0.237                | 0.289                | 0.306 [0.297, 0.315] | 0.263 [0.222, 0.303] |
| Modelo solo E_hull           | 0.299 [0.296, 0.302] | 0.299 [0.297, 0.300] | 0.213                | 0.237                | 0.289                | 0.306 [0.297, 0.315] | 0.263 [0.222, 0.303] |
| Modelo solo simetría         | 0.295 [0.291, 0.298] | 0.295 [0.292, 0.298] | 0.200                | 0.254                | 0.289                | 0.288 [0.275, 0.301] | 0.267 [0.253, 0.281] |
| Balance de carga (ICSD)      | 0.143                | 0.143                | 0.144                | 0.123                | 0.135                | 0.151 [0.146, 0.155] | 0.139 [0.135, 0.143] |
| Azar                         | 0.088 [0.075, 0.101] | 0.099 [0.087, 0.111] | 0.105 [0.096, 0.115] | 0.100 [0.084, 0.117] | 0.101 [0.089, 0.114] | 0.096 [0.084, 0.109] | 0.088 [0.064, 0.112] |

## 5. Sensibilidades

TPR:

| línea base                 |   random |   group_system |   loeo_A |   loeo_B |   family |   cluster |   temporal |
|:---------------------------|---------:|---------------:|---------:|---------:|---------:|----------:|-----------:|
| t en [0,8; 1,0] (clásica)  |    0.86  |          0.86  |    0.86  |    0.86  |    0.86  |     0.86  |      0.778 |
| E_hull = 0                 |    0.472 |          0.472 |    0.472 |    0.472 |    0.472 |     0.472 |      0.259 |
| E_hull ≤ 0,1               |    0.933 |          0.933 |    0.933 |    0.933 |    0.933 |     0.933 |      0.852 |
| Balance de carga (comunes) |    0.689 |          0.689 |    0.689 |    0.689 |    0.689 |     0.689 |      0.778 |

P(ŷ = 1 | U):

| línea base                 |   random |   group_system |   loeo_A |   loeo_B |   family |   cluster | temporal             |
|:---------------------------|---------:|---------------:|---------:|---------:|---------:|----------:|:---------------------|
| t en [0,8; 1,0] (clásica)  |    0.536 |          0.536 |    0.536 |    0.536 |    0.536 |     0.536 | 0.525 [0.484, 0.566] |
| E_hull = 0                 |    0.055 |          0.055 |    0.055 |    0.055 |    0.055 |     0.055 | 0.057 [0.041, 0.073] |
| E_hull ≤ 0,1               |    0.21  |          0.21  |    0.21  |    0.21  |    0.21  |     0.21  | 0.224 [0.201, 0.248] |
| Balance de carga (comunes) |    0.347 |          0.347 |    0.347 |    0.347 |    0.347 |     0.347 | 0.358 [0.327, 0.389] |

## 6. Lectura (descriptiva; las pruebas formales son de F4)

- **Solo simetría (control de procedencia de H3):** en `random`, TPR 0.896 con
  P(ŷ=1|U) = 0.126. El grupo espacial de la entrada de MP casi basta para separar P de U:
  las P son en su mayoría estructuras experimentales distorsionadas y las U, prototipos cúbicos
  hipotéticos. Es la procedencia de la estructura, no la sintetizabilidad. Fija la referencia
  que el modelo de grafos debe superar en H3.
- **Solo E_hull:** TPR 0.895 con P(ŷ=1|U) = 0.154 en `random`. Confirma la circularidad
  de la E_hull que se vio en F2.3.
- **τ:** TPR 0.922 con P(ŷ=1|U) = 0.415 (in-sample en la TableS1). Predice
  perovskita en muchas más U que los modelos con información de MP.
- **Caída máxima de TPR de `random` a un esquema confirmatorio**, en las líneas base
  aprendidas: E_hull 0.002 y simetría 0.003. Son
  modelos de una o dos entradas, que apenas dependen de la química del grupo reservado.
- **Azar:** TPR ≈ P(ŷ=1), como se espera. Es la referencia inferior de todas las curvas.
- **Cuidado al leer la sección 4:** en las reglas, la TPR a P(ŷ=1) fija cambia entre esquemas
  aunque cada composición reciba siempre la misma predicción. No es extrapolación: la P(ŷ=1)
  se fija dentro de cada fold, y la proporción de P de los folds cambia con el esquema. Por
  eso las comparaciones entre modelos a igual P(ŷ=1) se hacen pareadas, fold a fold (§7.2).
- **Puntuaciones discretas:** el modelo de solo simetría da pocas puntuaciones distintas (una
  por combinación de categorías), así que su umbral calibrado cae en bloques empatados y su
  P(ŷ=1|U) varía más entre semillas.
