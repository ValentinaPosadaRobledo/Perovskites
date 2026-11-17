# Informe F3.5, S2: fuga entre polimorfos en la partición de Gu et al. (2022)

> Generado por `scripts/report_f3_5_s2.py` (commit `56d418b`). No editar a mano.
> Modelos terminados en las dos variantes: 100 de 100. Secundario (§10b), no confirmatorio.

## Diseño

- **(a)** sus 100 particiones por cristal; **(b)** las mismas, con el mismo test, sin ningún
  cristal de train/val cuya composición esté en el test. Ambas reentrenadas en el entorno
  aislado (CPU, mismo código, semilla 0 de `configs/seeds.yaml`).
- h(c) = fracción de modelos que predicen positiva la positiva c (umbral 0,5); ΔTPR pareada
  por cristal, IC bootstrap jerárquico por composición (B = 10 000).
- 26 de las 94 positivas de test tienen un polimorfo en train/val (partición 0):
  la fuga solo puede actuar en ellas; las demás son el control.

## Resultados

| positivas de test          |   n |   TPR (a) por cristal |   TPR (b) agrupada |   ΔTPR (a − b) | IC 95 %          |   TPR ensemble (a) |   TPR ensemble (b) |
|:---------------------------|----:|----------------------:|-------------------:|---------------:|:-----------------|-------------------:|-------------------:|
| todas                      |  94 |                 0.912 |              0.907 |          0.004 | [-0.001, +0.010] |              0.957 |              0.957 |
| con polimorfo en train/val |  26 |                 0.891 |              0.878 |          0.013 | [-0.002, +0.032] |              0.923 |              0.923 |
| sin polimorfo en train/val |  68 |                 0.920 |              0.919 |          0.001 | [-0.002, +0.005] |              0.971 |              0.971 |

P(ŷ=1) del ensemble en las no etiquetadas de test: (a) 0.125, (b) 0.119.
