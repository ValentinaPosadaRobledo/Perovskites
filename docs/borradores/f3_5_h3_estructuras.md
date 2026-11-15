# H3 (F3.5): qué estructura usar por composición — análisis para decidir ANTES de entrenar

Datos: entradas de MP de `train_pool` (F1.5), sin entrenar nada. Por criterio, estructura
elegida para cada composición y su procedencia:

| Criterio | Clase | n | Cúbica | Coincide con el prototipo cúbico | B–O–B mín. (mediana) | Experimental |
|---|---|---|---|---|---|---|
| C1 perovskita de menor E_hull (= `post_dft` y línea base de simetría) | P | 193 | 9,8 % | 85,5 % | 148,7° | 97,4 % |
| | U | 619 | 88,0 % | 94,0 % | 180,0° | 0 % |
| C2 entrada de menor E_hull (cualquier clase) | P | 193 | 6,2 % | 75,1 % | 147,8° | 94,3 % |
| | U | 731 | 71,0 % | 75,4 % | 180,0° | 0 % |
| C3 P: perovskita experimental; U: perovskita teórica | P | 193 | 12,4 % | 85,5 % | 148,8° | 100 % |
| | U | 619 | 88,0 % | 94,0 % | 180,0° | 0 % |
| C4 solo perovskitas teóricas | P | 60 | 55,0 % | 98,3 % | 180,0° | 0 % |
| | U | 619 | 88,0 % | 94,0 % | 180,0° | 0 % |

**Lectura.** Con C1–C3 la procedencia separa casi por completo P de U: las P son estructuras
experimentales inclinadas (B–O–B ≈ 149°), las U prototipos cúbicos ideales (180°). Un modelo de
grafos puede aprender la distorsión, no la sintetizabilidad. Es lo mismo que mostró F2.4 (solo
simetría ≈ solo E_hull) y lo que la regla preregistrada de H3 anticipa (si la línea base de
simetría alcanza ≥ 50 % de la ventaja del grafo en `random`, la ventaja se atribuye a la
procedencia). C4 elimina la diferencia, pero solo cubre 60 de las 193 P (31 %), con un sesgo de
selección desconocido.

## Opciones

- **A (preregistrada):** C1, la misma entrada que `post_dft` y que la línea base de solo
  simetría, para que el control de procedencia sea directo. Cobertura 193 P y 587–619 U. Es
  probable que la regla de interpretación se active: entonces H3 se informa como "ventaja
  atribuible a la procedencia".
- **B (A + control sin procedencia, exploratorio):** además, el mismo modelo de grafos sobre
  **prototipos cúbicos ideales generados para TODAS las composiciones** (A en el vértice, B en
  el centro, O en las caras; parámetro de red de los radios iónicos o del prototipo de MP
  cuando existe). Elimina la procedencia con cobertura completa: mide si el grafo aporta algo
  más allá de la composición. Se declara como análisis exploratorio añadido (no estaba en el
  preregistro) y se decide antes de ver resultados.
- **C:** C4 (solo teóricas) como sensibilidad: sin procedencia, pero con 31 % de las P.

**Recomendación:** B. Mantiene el análisis preregistrado (A) y añade el control que de verdad
puede separar "estructura" de "procedencia". Coste extra: un segundo juego de entrenamientos
del grafo, como el de A (ver la decisión de cómputo pendiente: GPU o CPU con menos bolsas).

## Lo mismo en los datos de Gu (análisis rápido propuesto)

Sus positivas son estructuras de ICSD (experimentales) y sus no etiquetadas, sobre todo
hipotéticas (OQMD/AFLOW/MP). Propuesta: medir la fracción cúbica y el ángulo B–O–B de sus P y
U, y entrenar una línea base de solo grupo espacial en su partición. Si se acerca a su TPR de
0,957 a la misma P(ŷ=1), parte de su rendimiento publicado se explicaría por la procedencia.
Coste: análisis de simetría de 12 907 CIF (~10–40 min de CPU con prioridad baja).
