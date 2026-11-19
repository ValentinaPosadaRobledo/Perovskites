# F3.5 (exploratorio): ¿qué explica el rendimiento publicado de Gu et al.?

> Análisis exploratorio, **no preregistrado** (2026-11-19, aprobado por la autora antes de ver
> resultados). Sirve para decidir el criterio de estructura de H3; no es una prueba de hipótesis.

## Pregunta

Gu et al. (2022) obtienen TPR 0,957 en su test de 94 positivas. Sus positivas proceden de ICSD
y sus no etiquetadas son, sobre todo, estructuras hipotéticas. ¿Cuánto de ese TPR puede
alcanzarse **sin el modelo de grafos**, solo con lo que distingue la procedencia de los datos?

## Método

- **Datos:** sus 12 907 cristales (943 P, 11 964 U) y sus 100 particiones (mismo PTest de 94 P
  en todas). Rasgos por cristal calculados desde sus CIF en el entorno aislado
  (`isolated/gu2022/scripts/provenance_features.py`): grupo espacial (symprec 0,1 Å), Pm-3m y
  ángulo B–X–B mínimo. 0 errores de lectura. Solo se guardan identificadores y rasgos
  (`features.csv`), nunca estructuras.
- **Protocolo, el de su ensemble** (`provenance_report.py`): un modelo por partición, entrenado
  con PTrain + PVal frente a NTrain + NVal; puntuación = media de los 100. Evaluación en PTest
  frente a todas las U, igual que su ensemble (sus U de entrenamiento también cuentan en
  P(ŷ=1|U), en las mismas condiciones para todos los modelos). Con empates, TPR interpolado
  en la ROC (valor esperado al desempatar al azar). Semilla 0 de `configs/seeds.yaml`.
- **Comprobación:** su ensemble (`predict/Perov_All.json`) da TPR 0,957 en PTest a umbral 0,5,
  la cifra publicada; a ese umbral, P(ŷ=1|U) = 0,094.

## Descriptivo

| Clase | n | Pm-3m | B–X–B ≥ 179° | X = O o F |
|---|---|---|---|---|
| P (todas) | 943 | 68,4 % | 68,8 % | 39,1 % |
| PTest | 94 | 60,6 % | 62,8 % | 37,2 % |
| U (todas) | 11 964 | 87,6 % | 87,7 % | 90,9 % |

A diferencia de nuestros datos de MP (P inclinadas, U cúbicas; ver
`docs/borradores/f3_5_h3_estructuras.md`), en Gu la mayoría de las P también son cúbicas
Pm-3m. La diferencia más fuerte entre clases no es estructural, sino **química**: el 61 % de
sus P tiene en el sitio "X" un elemento que no es O ni F (P, Rh, Mn, Pd, Ca…: intermetálicos y
antiperovskitas), frente al 9 % de sus U.

## Resultado: TPR en PTest a igual P(ŷ=1|U)

| Modelo (protocolo de 100 particiones) | AUC (cota inferior) | TPR a 0,094 (su punto) | TPR a 0,05 | TPR a 0,10 | TPR a 0,20 |
|---|---|---|---|---|---|
| **Su ensemble de GCNN** | 0,982 | **0,957** | 0,936 | 0,957 | 0,968 |
| Regla "no Pm-3m" | 0,635 | 0,298 | 0,159 | 0,318 | 0,446 |
| Logística, solo grupo espacial | 0,637 | 0,361 | 0,209 | 0,373 | 0,442 |
| RF, grupo espacial + Pm-3m + ángulo B–X–B | 0,661 | 0,394 | 0,245 | 0,395 | 0,465 |
| Logística, solo el elemento X | 0,819 | 0,629 | 0,601 | 0,633 | 0,686 |
| Logística, solo los elementos A, B y X | 0,946 | 0,809 | 0,681 | 0,830 | 0,957 |

Su modelo acierta el 100 % de las 37 P de PTest no cúbicas y el 93 % de las 57 cúbicas; marca
como positivas el 34 % de las U no cúbicas y el 6 % de las cúbicas.

## Lectura

1. **La procedencia estructural (simetría y distorsión) explica poco en los datos de Gu:**
   resultado nulo para esa hipótesis. Solo con simetría, el TPR a su punto de operación es
   0,30–0,39, frente a 0,957.
2. **La química de la construcción del conjunto explica mucho:** solo con la identidad de los
   elementos A, B y X, sin estructura, se llega a 0,81 a su P(ŷ=1|U) (0,96 a 0,20); solo con
   el elemento X, a 0,63. La ventaja del grafo sobre la composición a igual P(ŷ=1|U) es de
   ≈ 0,15 de TPR, no la diferencia entre 0,957 y un azar.
3. Esto no invalida su modelo, pero matiza su comparación con métodos de composición: parte de
   la separación viene de que P y U se construyeron con químicas distintas.

## Implicaciones para H3 (nuestros datos)

En nuestro conjunto todas las composiciones son óxidos ABO₃: el sesgo químico de X no
existe. El que sí existe es el estructural (P experimentales inclinadas frente a U prototipos
cúbicos), que en Gu es débil. Por eso este análisis **no cambia** la recomendación del
borrador: la opción B (A + control con prototipos cúbicos para todas las composiciones) sigue
siendo la que separa estructura de procedencia en nuestros datos.

## Limitaciones

- Exploratorio y no preregistrado; una sola semilla para los modelos de RF (los de regresión
  logística son deterministas).
- P(ŷ=1|U) incluye U usadas en entrenamiento (como en su ensemble): las cifras de todos los
  modelos son optimistas por igual.
- El ángulo B–X–B usa un corte de enlace de 1,2 veces la distancia B–X mínima; en las
  estructuras que no son redes de vértices compartidos, es solo un descriptor.
