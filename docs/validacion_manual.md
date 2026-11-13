# Validación manual de la identificación de perovskitas (F1.4)

Objetivo: revisar a mano **al menos 50 casos límite** del clasificador por conectividad
(regla 2.1.1 de `CLAUDE.md`) y documentar cada decisión. Los criterios y tolerancias están
en `configs/data.yaml` (sección `identification`).

> **Licencia ICSD:** no copiar aquí estructuras ni coordenadas de ICSD; solo identificadores
> (`material_id` de MP, ID de ICSD) y descriptores derivados.

## Categorías de casos límite a cubrir

Al menos unos pocos casos de cada categoría:

1. Perovskitas ortorrómbicas/romboédricas muy distorsionadas (ángulo B–O–B cerca del umbral).
2. Politipos hexagonales 2H/4H/6H (octaedros que comparten caras) → clase aparte.
3. Estructuras tipo ilmenita (FeTiO₃) y LiNbO₃ → no perovskita.
4. YMnO₃ hexagonal (B en coordinación bipiramidal trigonal) → no perovskita.
5. Post-perovskita (CaIrO₃, MgSiO₃ ppv: octaedros que comparten aristas) → no perovskita.
6. Polianiones con fórmula ABO₃ (carbonatos, nitratos, boratos, yodatos) → excluidos.
7. Estados de oxidación ambiguos (BiNiO₃, CaFeO₃) → bandera.
8. Fases de alta presión/metaestables (MgSiO₃, BiCoO₃, PbVO₃, BiScO₃) → bandera.
9. Desacuerdos entre CrystalNN y `StructureMatcher`.
10. Casos con A pequeño (NC del sitio A < 8) o B no octaédrico.

## Candidatos identificados en F1.2

Conflictos de etiqueta entre Bartel et al. (2019) y Gu et al. (2022). La lista completa,
generada, está en `results/f1_2_datos_externos/conflicts.csv`.

- **Sintetizados NO perovskita para Bartel, positivos para Gu (16):** CdCO₃, CoSeO₃, CoTeO₃,
  CsBrO₃, CuSeO₃, CuTeO₃, KBrO₃, MgSeO₃, NiSeO₃, RbBrO₃, RbClO₃, SrSiO₃, FeTiO₃ (TiFeO₃ en
  Gu), MnTiO₃ (TiMnO₃ en Gu), TlBrO₃, ZnSeO₃. Corresponden a las categorías 3 y 6.
- **Perovskitas para Bartel, solo virtuales en Gu (41):** p. ej. BaHfO₃, CaHfO₃, CeFeO₃,
  DyTiO₃. Comprobar si nuestra identificación las reconoce y por qué Gu no las tiene como
  positivas.

## Registro

La planilla está en **`docs/validacion_manual_casos.csv`**: 59 casos, generados por
`scripts/prepare_f1_4.py` a partir de la identificación de F1.3. Se abre directamente en
Excel o en Google Sheets.

**Cómo revisar cada caso**

1. Busca la estructura en Materials Project con el ID (`mp_id_antiguo`, p. ej. `mp-5229`, o
   `material_id` en el formato nuevo) y mira su visualización: poliedros BO₆, cómo se unen y
   dónde está el catión A.
2. Lee `que_mirar` (la pregunta clave de su categoría) y las métricas automáticas: NC del
   sitio A, O a < 3,3 Å del A, compartición máxima de los octaedros y ángulo B–O–B mínimo.
3. Rellena las columnas vacías:
   - `clasificacion_manual`: `perovskita`, `politipo_hexagonal`, `no_perovskita`, `polianion`
     o `dudosa`;
   - `coincide`: `si` o `no`, comparando con `clasificacion_automatica` (un caso
     `perovskite_borderline` coincide si lo clasificas como `perovskita`);
   - `motivo_referencia`: una frase, con la referencia bibliográfica si la usas;
   - `revisora` y `fecha`.
4. Se puede hacer por tandas. Cuando termines, la concordancia y los ajustes del criterio se
   calculan con un script (pendiente: `scripts/report_f1_4.py`).

**Observaciones previas de la preparación (a confirmar en la revisión)**

- Casos 47–51 (NC(A) = 0): estructuras cúbicas marcadas como experimentales en MP en las que
  el catión **grande** ocupa el sitio octaédrico (p. ej. La en el sitio B de LaGaO₃). Podrían
  ser asignaciones de sitio invertidas en MP. El algoritmo no las acepta como perovskitas.
- Casos 4 y 8 (AgSbO₃, KSbO₃, Fd-3m): probablemente pirocloros, una red de vértices que no es
  perovskita. El algoritmo los deja como caso límite por la NC(A) = 6.
- Caso 46 (PbVO₃ P4mm): V en pirámide de base cuadrada (VO₅). La fase de alta presión
  conocida es una perovskita tetragonal muy distorsionada: decidir si cuenta.

## Resumen

- Casos preparados: 59 (≥ 50). Revisados: 0
- Desacuerdos: —
- Ajustes de criterios derivados (registrar también en la bitácora del ROADMAP): —
