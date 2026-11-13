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

| # | material_id | Fórmula | Grupo espacial | NC(B) | NC(A) | Compartición BO₆ | Ángulo B–O–B mín (°) | Clasificación automática | Clasificación manual | ¿Coinciden? | Motivo / referencia | Revisora | Fecha |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | | | | | | | | | | | | | |

## Resumen

- Casos revisados: 0 / 50
- Desacuerdos: —
- Ajustes de criterios derivados (registrar también en la bitácora del ROADMAP): —
