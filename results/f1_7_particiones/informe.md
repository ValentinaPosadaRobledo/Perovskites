# Informe F1.7: particiones

> Generado por `scripts/report_f1_7.py` (commit `a347eb5`).
> No editar a mano. **Pool: train_pool (definitivo, tras congelar el holdout): 924 composiciones, 193 P.**
> Las particiones definitivas usan solo `train_pool` (regla 2.1.5); están en
> `data/splits/folds.parquet`, con su hash en `HASHES.json`.

## 1. Implementación

Implementación propia en `src/perovskite_pu/splits/` y **no MatFold** (motivo en la bitácora):
MatFold no tiene particiones por sitio A/B (`loeo_A`/`loeo_B`), por familia del elemento B, ni
LOCO por k-means o temporal, y exige una estructura por entrada, mientras que nuestras
etiquetas son por composición. Lo que MatFold sí ofrece (`random`, `chemsys`) coincide con
nuestro `random` y `group_system`.

## 2. Esquemas (semilla 0)

| esquema      |   folds |   grupos | tamaño de grupo (mín/mediana/máx)   | tamaño de test (mín/mediana/máx)   | P en test (mín/mediana/máx)   |   folds sin P |
|:-------------|--------:|---------:|:------------------------------------|:-----------------------------------|:------------------------------|--------------:|
| random       |       5 |      nan | —                                   | 184 / 185 / 185                    | 31 / 39 / 44                  |             0 |
| group_system |       5 |      924 | 1 / 1 / 1                           | 184 / 185 / 185                    | 35 / 38 / 43                  |             0 |
| loeo_A       |      69 |       69 | 1 / 9 / 46                          | 1 / 9 / 46                         | 0 / 0 / 16                    |            38 |
| loeo_B       |      16 |       16 | 13 / 33.5 / 218                     | 13 / 33.5 / 218                    | 1 / 13 / 21                   |             0 |
| family       |       4 |        4 | 141 / 244 / 295                     | 141 / 244 / 295                    | 13 / 36 / 108                 |             0 |
| cluster      |       5 |        5 | 5 / 175 / 373                       | 5 / 175 / 373                      | 0 / 23 / 121                  |             1 |
| temporal     |       1 |      nan | —                                   | 137 / 137 / 137                    | 27 / 27 / 27                  |             0 |

- **`loeo_B`:** los elementos B con < 5 positivas se fusionan por familia.
- **`loeo_A`:** se reportan aparte (`report_separately`); hay folds sin positivas, que solo
  aportan P(ŷ=1|U). Por eso la TPR se micro-agrega.

## 3. `group_system` frente a `random`

Tamaños de grupo de `group_system` (tamaño: n.º de grupos): {1: 924}. **En ABO₃, cada
sistema químico A–B contiene exactamente una composición**, así que `group_system` es,
por construcción, idéntico a `random`. Se mantiene solo como diagnóstico, como prevé la
regla 2.1.4.

## 4. `cluster`: silueta de k-means sobre MAGPIE estandarizado (10 semillas)

k=2: 0.166±0.000 · k=3: 0.157±0.001 · k=4: 0.158±0.006 · k=5: 0.163±0.004 · k=6: 0.149±0.012 · k=7: 0.147±0.010 · k=8: 0.146±0.004 · k=9: 0.151±0.003 · k=10: 0.155±0.004

La silueta es plana: no hay un k natural. Se usa **k = 5** (decisión de la autora).

## 5. `temporal`

Corte: **2009** (fracción objetivo 15%). Test: 27 positivas (14.0%) caracterizadas
desde 2009, más la misma fracción de U, elegida con la semilla. Limitación: la positiva más
reciente es de 2013, porque la copia de ICSD que usa MP no cubre años posteriores.

## 6. S1: ¿extrapolan los esquemas? (preregistro §10b)

Referencia `random`: mediana de la distancia al vecino de train más cercano
2.09; umbral (q95) 4.95. Espacio: MAGPIE estandarizado + PCA al 95 %,
ajustados en train.

| Esquema | Distancia mediana | Grupos extrapolativos | Positivas en grupos extrapolativos |
|---|---|---|---|
| `loeo_A` | 2.61 | 4% | 0% |
| `loeo_B` | 3.89 | 12% | 12% |
| `family` | 5.03 | 50% | 25% |
| `cluster` | 6.84 | 100% | 100% |

Interpretación preregistrada: si < 50 % de las positivas de un esquema están en grupos
extrapolativos, sus resultados de H1 se describen como de "extrapolación débil".

### Sensibilidad 10: espacio estandarizado completo, sin PCA

Referencia `random`: mediana 2.51; umbral (q95) 5.75.
Las distancias no son comparables en valor absoluto con las de arriba (otra dimensión); sí lo
son las fracciones de grupos y de positivas en grupos extrapolativos.

| Esquema | Distancia mediana | Grupos extrapolativos | Positivas en grupos extrapolativos |
|---|---|---|---|
| `loeo_A` | 3.03 | 10% | 3% |
| `loeo_B` | 4.43 | 6% | 5% |
| `family` | 5.76 | 25% | 18% |
| `cluster` | 7.45 | 100% | 100% |
