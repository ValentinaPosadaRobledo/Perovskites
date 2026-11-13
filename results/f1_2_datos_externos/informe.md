# Informe F1.2: datos externos y su cruce con MP

> Generado por `scripts/report_f1_2.py` (commit `13fbe30`).
> No editar a mano. MP 2026.09.28. Cifras en `summary.json`.

## 1. Gu et al. (2022), Zenodo 6348981

| | Cristales | Positivos | Composiciones |
|---|---|---|---|
| Todas las clases | 12907 | 943 | 6611 |
| Óxidos ABO₃ | 7502 | 312 | 3203 (262 con algún positivo) |

**Fuga entre polimorfos en su test** (dato de partida de S2, preregistro §10b). El test de
positivos es idéntico en sus 100 particiones.

| Subconjunto | Positivos en test | Composición también en train/val | Composición entre positivos de train/val |
|---|---|---|---|
| Todas las clases | 94 | 26 (28%) | 21 (22%) |
| Óxidos ABO₃ | 30 | 11 (37%) | 6 (20%) |

## 2. Gu et al. frente a MP 2026

- **IDs:** MP pasó a IDs alfabéticos; se convirtieron al formato numérico de Gu
  (`to_legacy_mpid`). Cristales ABO₃ de Gu con ID de MP: 1182;
  encontrados en MP 2026: 1167.
- **Composiciones:** 1303 compartidas. Composiciones de nuestra
  descarga ausentes en Gu: 16. Gu reúne además OQMD y AFLOW, así
  que tiene muchas más composiciones.
- **Etiquetas por composición** (compartidas): Gu positivo y MP experimental:
  249. Gu virtual pero MP experimental:
  **191**. Gu positivo pero MP solo teórica:
  7.
- Las "Gu virtual, MP experimental" pueden ser polimorfos sintetizados **no perovskita** (la
  categoría aparte del proyecto) o síntesis posteriores a 2020. Se distinguen en F1.3.

## 3. Bartel et al. (2019), TableS1 (commit `965b8de`)

576 compuestos; **369 óxidos ABO₃** (232 perovskita,
137 no perovskita); 359 están en MP 2026.

**Frente a MP 2026:**

| Bartel                |   MP: experimental |   MP: solo teórica |   no está en MP |   Total |
|:----------------------|-------------------:|-------------------:|----------------:|--------:|
| Bartel: NO perovskita |                128 |                  8 |               1 |     137 |
| Bartel: perovskita    |                189 |                 34 |               9 |     232 |
| Total                 |                317 |                 42 |              10 |     369 |

**34 perovskitas experimentales según Bartel figuran en MP solo
como teóricas:** con nuestra definición quedarían como no etiquetadas. Es una medida directa
de positivos ocultos en U.

**Frente a Gu et al.:**

| Bartel                |   Gu: positivo |   Gu: virtual |   no está en Gu |   Total |
|:----------------------|---------------:|--------------:|----------------:|--------:|
| Bartel: NO perovskita |             16 |           121 |               0 |     137 |
| Bartel: perovskita    |            188 |            41 |               3 |     232 |
| Total                 |            204 |           162 |               3 |     369 |

## 4. Conflictos de etiqueta entre Bartel y Gu (`conflicts.csv`)

- **Bartel: NO perovskita; Gu: positivo (16):**
  CdCO3, CoSeO3, CoTeO3, CsBrO3, CuSeO3, CuTeO3, KBrO3, MgSeO3, NiSeO3, RbBrO3, RbClO3, SrSiO3, TiFeO3, TiMnO3, TlBrO3, ZnSeO3
- **Bartel: perovskita; Gu: solo virtual (41):**
  BaHfO3, CaHfO3, CaUO3, CeFeO3, CeLuO3, CeTiO3, CeTmO3, DyRhO3, DyTiO3, ErAlO3, ErRhO3, ErTiO3, EuGaO3, GdRhO3, HoRhO3, HoTiO3, KPaO3, LaCuO3, LaNiO3, LuTiO3, NaIO3, NaPaO3, NdInO3, PrTiO3, PuCrO3, PuVO3, RbNbO3, RbPaO3, RbTaO3, SmInO3, SrIrO3, SrPuO3, TaAgO3, TbTiO3, TiHgO3, TmAlO3, TmTiO3, YGaO3, YbCeO3, YbPrO3, YbTiO3

El primer grupo incluye polianiones (carbonatos, selenitos, teluritos, bromatos, cloratos,
silicatos) e ilmenitas (FeTiO₃, MnTiO₃), que no son perovskitas. Es coherente con que la
identificación por similitud a un prototipo acepte falsos positivos, lo que motiva la regla
2.1.1. **Es un conflicto entre fuentes, no un error demostrado:** se verifica con las
estructuras en F1.3/F1.4, donde estos compuestos entran como casos límite.

## 5. Implicaciones

1. **Formabilidad (F3.4):** antes de usar los negativos de Bartel hay que excluir los que
   nuestra identificación marque como perovskita (regla 2.3). Los 16
   conflictos del primer grupo se revisan a mano.
2. **PU:** las 34 perovskitas de Bartel que en MP son solo
   teóricas permiten estimar cuántos positivos hay escondidos en U. Sirve de contraste para la
   prior π (F3.2).
3. **S2 (F3.5):** la fuga entre polimorfos en el test de Gu es sustancial
   (28% en todas las clases), lo que confirma que S2 es necesario.
