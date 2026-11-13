# Informe F1.3: identificación de perovskitas por conectividad

> Generado por `scripts/report_f1_3.py` (commit `a2f0a73`).
> No editar a mano. Cifras en `summary.json`; casos límite en `borderline.csv`.

## 1. Criterio aplicado (regla 2.1.1 de CLAUDE.md, decisiones del 2026-11-13)

Una entrada es **perovskita** si sus octaedros BO₆ comparten **solo vértices**, cada O une
exactamente 2 B, el ángulo B–O–B mínimo es ≥ **135°** y la NC del sitio A
(CrystalNN) está entre 8 y 12. Excepción para A con par libre
(Bi, Pb, Sn, Sb, Tl, Hg): basta con ≥ 8 O a < 3.3 Å.
Sensibilidad: la regla original (140°, sin excepción).

## 2. Resultado

| Clase | Entradas | Entradas experimentales |
|---|---|---|
| perovskite | 1244 | 329 |
| non_perovskite | 611 | 194 |
| polyanion | 390 | 210 |
| perovskite_borderline | 255 | 65 |
| hexagonal_polytype | 63 | 31 |

**Composiciones con al menos una entrada experimental perovskita: 223**
(199 con la regla de sensibilidad). Es un avance de F1.5, que fija
formalmente las etiquetas.

**Estados de oxidación:** BVAnalyzer en 1754 entradas, conjetura por composición en
188, ninguno en 231 (no polianiones; se usó CrystalNN sin
restricción catión–anión). Marcadas como ambiguas: 683.

## 3. Conectividad frente a similitud con el prototipo cúbico

| cls                   |   coincide con prototipo |   no coincide |
|:----------------------|-------------------------:|--------------:|
| hexagonal_polytype    |                        0 |            63 |
| non_perovskite        |                        2 |           609 |
| perovskite            |                     1164 |            80 |
| perovskite_borderline |                       84 |           171 |

- **80 entradas perovskita no coinciden con el prototipo cúbico** (StructureMatcher, con
  las tolerancias por defecto): una identificación solo por prototipo las perdería.
- **86 entradas que NO son perovskita según la conectividad sí coinciden** con el
  prototipo: serían falsos positivos de esa identificación.

## 4. Validación frente a Bartel et al. (2019), *in-sample*

> **Circularidad declarada:** el umbral de 135° y la regla del par libre se eligieron
> comparando con estas etiquetas. Esta tabla NO es una validación independiente; la
> independiente es la revisión manual de F1.4.

| Regla | Composiciones comparadas | Perovskitas detectadas | No perovskitas mal marcadas |
|---|---|---|---|
| Principal (135°, híbrida) | 317 | 176/189 (93.1%) | 2 (1.6%): SrSiO3, TiMnO3 |
| Sensibilidad (140°, estricta) | 317 | 158/189 (83.6%) | 2 (1.6%): SrSiO3, TiMnO3 |

- **Falsos positivos:** SrSiO₃ y MnTiO₃ (TiMnO₃) tienen en MP entradas experimentales con
  estructura de perovskita; Bartel los clasifica como no perovskita. Que sean polimorfos de
  alta presión es plausible, pero **no está verificado**: se comprueba en F1.4 y, si se
  confirma, reciben la bandera de alta presión (F1.5).
- **No detectadas (13):** CsIO3, KIO3, LaErO3, LaHoO3, LaYO3, NaIO3, RbIO3, RbNbO3, RbTaO3, ScBiO3, ScCrO3, TlIO3, YGaO3. Incluyen yodatos (excluidos por
  definición: el I⁵⁺ forma pirámides IO₃, no octaedros); compuestos cuyo polimorfo
  perovskita no figura como experimental en MP (RbNbO₃, RbTaO₃, YGaO₃), y perovskitas muy
  distorsionadas a cuyo sitio A CrystalNN asigna NC 6 (LaErO₃, LaHoO₃, LaYO₃, ScCrO₃,
  ScBiO₃). Estas últimas quedan como casos límite para la revisión manual.

## 5. Conflictos Bartel–Gu (F1.2) según nuestra identificación

| type                 | nuestra_clase                  |   n |
|:---------------------|:-------------------------------|----:|
| bartel_no_gu_si      | non_perovskite                 |   1 |
| bartel_no_gu_si      | perovskite                     |   2 |
| bartel_no_gu_si      | perovskite_borderline          |   2 |
| bartel_no_gu_si      | polyanion                      |  11 |
| bartel_si_gu_virtual | non_perovskite                 |   3 |
| bartel_si_gu_virtual | perovskite                     |   5 |
| bartel_si_gu_virtual | polyanion                      |   1 |
| bartel_si_gu_virtual | sin entrada experimental en MP |  32 |

## 6. Casos límite → validación manual (F1.4)

255 entradas (65 experimentales) en `borderline.csv`, ordenadas
con las experimentales primero.
