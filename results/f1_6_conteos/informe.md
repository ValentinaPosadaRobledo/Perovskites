# Informe F1.6: conteos

> Generado por `scripts/report_f1_6.py` (commit `1d11732`).
> No editar a mano. Tablas completas: `counts_by_A.csv`, `counts_by_B.csv`, `counts_by_family.csv`.

## 1. Punto de decisión del ROADMAP

**226 composiciones ABO₃ positivas** (umbral: ~100; estimación previa: 200–350). **Decisión:
mantener alcance ABO3 óxidos**, sin ampliar a otras clases de perovskita.

## 2. Etiquetas

| Etiqueta | Composiciones |
|---|---|
| unlabeled | 824 |
| positive | 226 |
| polyanion | 160 |
| synthesized_non_perovskite | 84 |
| ambiguous | 25 |

- **Conjunto PU: 1050**; prevalencia de positivas etiquetadas **21.5%**.
- Sensibilidad, solo con U que admiten balance de carga: PU = 740; prevalencia 30.5%.

## 3. Estructura de las perovskitas experimentales (332 entradas)

**Sistema cristalino:** Orthorhombic 190, Cubic 81, Trigonal 26, Monoclinic 18, Tetragonal 17.

**Grupos espaciales más frecuentes:** Pnma (175), Pm-3m (79), R-3c (18), I4/mcm (10), P2_1/c (9), C2/c (4), P4mm (4), R3m (3), P4/mbm (3), R3c (3).

**Todas las entradas por clase final:**

| cls_final             |   experimental |   teórica |
|:----------------------|---------------:|----------:|
| doubtful              |              8 |         0 |
| hexagonal_polytype    |             31 |        32 |
| non_perovskite        |            200 |       417 |
| perovskite            |            332 |       915 |
| perovskite_borderline |             48 |       190 |
| polyanion             |            210 |       180 |

## 4. Por familia del elemento B

| family     |   P |   U |   sint. no perov. |   ambigua |   total_PU |
|:-----------|----:|----:|------------------:|----------:|-----------:|
| 3d         | 124 | 175 |                29 |         9 |        299 |
| 4d_5d      |  46 | 224 |                14 |         8 |        270 |
| 4f_5f      |  15 | 135 |                 3 |         4 |        150 |
| main_group |  41 | 290 |                38 |         4 |        331 |

## 5. Por elemento B (15 con más positivas; 72 elementos B en P ∪ U)

| B   |   P |   U |   sint. no perov. |   ambigua |   total_PU |
|:----|----:|----:|------------------:|----------:|-----------:|
| Cr  |  20 |  14 |                 1 |         1 |         34 |
| V   |  19 |  15 |                 9 |         1 |         34 |
| Fe  |  18 |  12 |                 3 |         0 |         30 |
| Mn  |  16 |  15 |                 5 |         0 |         31 |
| Al  |  13 |  25 |                 0 |         0 |         38 |
| Ti  |  13 |  24 |                 4 |         2 |         37 |
| Ni  |  13 |  20 |                 2 |         1 |         33 |
| Co  |  13 |  19 |                 3 |         1 |         32 |
| Sc  |  11 |  15 |                 0 |         2 |         26 |
| Rh  |   9 |  18 |                 1 |         2 |         27 |
| Ga  |   7 |  13 |                 2 |         0 |         20 |
| Nb  |   7 |   6 |                 3 |         1 |         13 |
| Sn  |   5 |  14 |                 2 |         1 |         19 |
| Zr  |   5 |  12 |                 0 |         0 |         17 |
| Ge  |   4 |  21 |                 4 |         0 |         25 |

## 6. Por elemento A (15 con más positivas; 70 elementos A en P ∪ U)

| A   |   P |   U |   sint. no perov. |   ambigua |   total_PU |
|:----|----:|----:|------------------:|----------:|-----------:|
| Sr  |  21 |  31 |                 1 |         1 |         52 |
| Ba  |  18 |  24 |                10 |         0 |         42 |
| La  |  18 |  12 |                 1 |         2 |         30 |
| Ca  |  16 |  21 |                 2 |         0 |         37 |
| Pr  |  13 |   7 |                 0 |         0 |         20 |
| Nd  |  11 |  13 |                 0 |         0 |         24 |
| Eu  |  10 |  36 |                 0 |         0 |         46 |
| Sm  |  10 |  18 |                 0 |         0 |         28 |
| Gd  |   9 |  14 |                 0 |         0 |         23 |
| Dy  |   9 |  13 |                 0 |         0 |         22 |
| Ho  |   9 |  13 |                 0 |         0 |         22 |
| Y   |   9 |  13 |                 0 |         0 |         22 |
| Tb  |   8 |  15 |                 0 |         0 |         23 |
| Bi  |   7 |  18 |                 1 |         2 |         25 |
| Na  |   6 |  26 |                 5 |         0 |         32 |

## 7. Avance para F1.7

- Elementos **B** con ≥ 5 positivas: **14**. Con menos, se fusionan por familia
  (`small_group_policy: merge_family`): 27 elementos B con entre 1 y 4 positivas.
- Elementos **A** con ≥ 5 positivas: **21** (los demás se reportan aparte,
  `report_separately`).
- `family` tiene 4 grupos; el holdout (F1.8) reserva elementos B estratificando por
  familia hasta cubrir ~10–15 % de las positivas.
