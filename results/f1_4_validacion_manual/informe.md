# Informe F1.4: validación manual de la identificación de perovskitas

> Generado por `scripts/report_f1_4.py` (commit `1cb0383`).
> No editar a mano. Revisora: Valentina Posada (2026-11-13). Casos: 59.

## 1. Concordancia

| clasificacion_automatica   |   dudosa |   no_perovskita |   perovskita |   polianion |   politipo_hexagonal |   Total |
|:---------------------------|---------:|----------------:|-------------:|------------:|---------------------:|--------:|
| hexagonal_polytype         |        0 |               0 |            0 |           0 |                    6 |       6 |
| non_perovskite             |        0 |              14 |            0 |           0 |                    0 |      14 |
| perovskite                 |        0 |               0 |           16 |           0 |                    0 |      16 |
| perovskite_borderline      |        8 |               6 |            3 |           0 |                    0 |      17 |
| polyanion                  |        0 |               0 |            0 |           6 |                    0 |       6 |
| Total                      |        8 |              20 |           19 |           6 |                    6 |      59 |

**Clasificaciones firmes del algoritmo** (perovskita, no perovskita, politipo hexagonal,
polianión) **confirmadas por la revisora: 42/42 (100%; IC 95 % Clopper–Pearson
91.6%–100.0%).** Esta es la validación independiente de la identificación: no comparte la
circularidad de la comparación con Bartel (F1.3).

## 2. Casos límite (17)

La categoría "límite" no se acepta como perovskita. La revisión confirma que funciona como
filtro: 3 de 17 resultaron perovskitas y el resto son no perovskitas
(tipo LiNbO₃, pirocloros) o dudosas.

| Caso | Fórmula | Grupo | ID MP | Veredicto | Motivo (revisora) |
|---|---|---|---|---|---|
| 1 | ZnSnO3 | R3c | `mp-13334` | no_perovskita | Tipo LiNbO3 (R3c): el Zn ocupa un sitio octaédrico 3+3 (2.03/2.36 Å), no la cavidad cuboctaédrica de una perovskita; solo los SnO6 comparten vértices. |
| 2 | InRhO3 | Pnma | `mp-1105819` | perovskita | MP la describe como perovskita ortorrómbica Pnma: los RhO6 comparten solo vértices (inclinación 42–45°) y el In ocupa la cavidad A con coordinación 6 (2.18–2.40 Å) más vecinos lejanos. |
| 3 | SrPrO3 | Pnma | `mp-20464` | dudosa | Los PrO6 comparten solo vértices (Pnma, inclinación 38–44°), pero el Sr solo tiene 6 O (2.53–2.79 Å), un O queda 4-coordinado y MP la marca como representación ordenada de una estructura desordenada. |
| 4 | AgSbO3 | Fd-3m | `mp-540872` | no_perovskita | Pirocloro defectuoso (Fd-3m): los SbO6 comparten vértices formando túneles y el Ag queda en coordinación plano-hexagonal 6 (2.56 Å); no hay cavidad A de perovskita. |
| 5 | LaYO3 | Pnma | `mp-10429` | perovskita | Perovskita Pnma: los YO6 comparten solo vértices (inclinación 40–44°) y el La está 8-coordinado (2.37–3.15 Å). |
| 6 | ScCrO3 | Pnma | `mp-18961` | perovskita | Perovskita Pnma: los CrO6 comparten solo vértices (inclinación 42–46°) y el Sc está 8-coordinado (2.08–2.75 Å). |
| 8 | KSbO3 | Fd-3m | `mp-16293` | no_perovskita | Pirocloro defectuoso (Fd-3m) como AgSbO3: los SbO6 comparten vértices, pero el K está en coordinación plano-hexagonal 6 (2.68 Å), no en una cavidad cuboctaédrica. |
| 18 | LiTaO3 | R3c | `mp-3666` | no_perovskita | Tipo LiNbO3: el Li ocupa un sitio octaédrico 3+3 (2.03/2.27 Å) y los TaO6 se inclinan 40°; el A no está en una cavidad cuboctaédrica. |
| 19 | LiNbO3 | R3c | `mp-3731` | no_perovskita | LiNbO3: el Li octaédrico comparte vértices, aristas y una cara con los NbO6; es el prototipo de la familia y no una perovskita. |
| 20 | NaNbO3 | R3c | `mp-4514` | no_perovskita | Tipo LiNbO3: el Na está en coordinación 6 (3+3, 2.37/2.51 Å) y los NbO6 se inclinan 27°; el A no está en una cavidad cuboctaédrica. |
| 41 | BiScO3 | Cc | `mp-555313` | dudosa | Los ScO6 comparten vértices (inclinación 22–36°), pero el Bi solo tiene 4–5 O y los O forman tetraedros con aristas compartidas: perovskita muy distorsionada. |
| 47 | GaPrO3 | Pm-3m | `mp-9833` | dudosa | Topología de perovskita cúbica (PrO6 de vértices sin inclinar), pero el Ga no tiene O vecinos (≈3.2 Å) y el Pr octaédrico a 2.30 Å: roles A/B invertidos. |
| 48 | GaLaO3 | Pm-3m | `mp-9831` | dudosa | Topología de perovskita cúbica (LaO6 de vértices sin inclinar), pero el Ga no tiene O vecinos (≈3.2 Å) y el La octaédrico a 2.30 Å: roles A/B invertidos. |
| 49 | PbBaO3 | Pm-3m | `mp-561598` | dudosa | Topología cúbica con BaO6 octaédricos a 2.34 Å (demasiado corto para Ba) y Pb sin O vecinos: roles A/B invertidos. |
| 50 | GaNdO3 | Pm-3m | `mp-9834` | dudosa | Topología de perovskita cúbica (NdO6 de vértices sin inclinar), pero el Ga no tiene O vecinos (≈3.2 Å) y el Nd octaédrico a 2.28 Å: roles A/B invertidos. |
| 51 | TiCdO3 | Pm-3m | `mp-22345` | dudosa | CdO6 de vértices sin inclinar y el Ti en cuboctaedro TiO12 a 2.90 Å: topología de perovskita, pero con el catión pequeño en el sitio A y roles invertidos. |
| 53 | TeCoO3 | Pnma | `mp-19113` | dudosa | Los CoO6 comparten solo vértices (Pnma, inclinación 47–52°), pero el Te4+ forma grupos TeO3 piramidales de 3 O: más parecido a un telurito que a una perovskita. |

## 3. Hallazgos

1. **Tipo LiNbO₃ frente a perovskita:** la NC del sitio A por distancias no los separa
   (NaNbO₃ R3c tiene 12 O a < 3,3 Å y no es perovskita). Justifica no haber adoptado la regla
   de distancias para todos los A (F1.3).
2. **Falsos negativos recurrentes:** perovskitas Pnma en las que CrystalNN da NC 6 al sitio A
   (InRhO₃, LaYO₃, ScCrO₃). **No se ajusta el criterio** con estos 59 casos, para evitar el
   sobreajuste; en los casos revisados prevalece el veredicto manual (decisión 2026-11-14).
3. **Entradas de MP con roles A/B invertidos** (LaGaO₃, PrGaO₃, NdGaO₃, BaPbO₃, CdTiO₃ en
   Pm-3m): catión grande en el sitio octaédrico. No afectan a las etiquetas, porque esas
   composiciones tienen otras entradas experimentales perovskita correctas.
4. **Pirocloros (AgSbO₃, KSbO₃):** red de octaedros con vértices compartidos que no es
   perovskita; el algoritmo los dejó correctamente como límite por la NC(A) = 6.

## 4. Efecto sobre las etiquetas por composición (vista previa de F1.5)

|                            |   antes de la revisión |   después |
|:---------------------------|-----------------------:|----------:|
| unlabeled                  |                    824 |       824 |
| positive                   |                    223 |       226 |
| polyanion                  |                    160 |       160 |
| synthesized_non_perovskite |                     82 |        84 |
| ambiguous                  |                     30 |        25 |

- Conjunto PU (P + U): **1050** composiciones. Prevalencia de positivas etiquetadas:
  **21.5%** (23.3% si las ambiguas contaran como P).
- **Ambiguas (25)**, excluidas del análisis principal (P en la sensibilidad):
  CoTeO3, CrInO3, KBiO3, KIrO3, KSbO3, LaErO3, LaHoO3, LiNbO3, LiOsO3, LiReO3, LiTaO3, LiUO3, LuRhO3, NiBiO3, OsPbO3, ScBiO3, ScMnO3, ScRhO3, SnPbO3, SrPrO3, TcPbO3, TiFeO3, TiZnO3, VCuO3, ZnPbO3.
