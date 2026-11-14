# Informe F1.5: etiquetas por composición

> Generado por `scripts/report_f1_5.py` (commit `846444d`).
> No editar a mano. Tabla completa: `labels.csv`.

## 1. Deduplicación estructural (StructureMatcher, dentro de cada composición)

2563 entradas → **2135 estructuras únicas** (257 grupos con duplicados,
428 entradas redundantes). Con las tolerancias por defecto, StructureMatcher agrupa
variantes distorsionadas del mismo armazón (p. ej. una perovskita cúbica teórica con su
versión R3c experimental). **Las etiquetas usan todas las entradas de cada composición**,
así que la deduplicación no las afecta. Solo importará si F3.5 necesita una estructura
representativa por composición.

"Agrupar composiciones dopadas o casi idénticas con su padre" no aplica: MP solo contiene
ABO₃ ordenados y estequiométricos.

## 2. Etiquetas (1319 composiciones)

| Etiqueta | Composiciones | Uso |
|---|---|---|
| `positive` (P) | 226 | positivos del problema PU |
| `unlabeled` (U) | 824 | no etiquetados (NO son negativos) |
| `synthesized_non_perovskite` | 84 | aparte: validación de formabilidad (F3.4) |
| `ambiguous` | 25 | fuera de P y U; P en la sensibilidad (F3.6) |
| `polyanion` | 160 | excluidas |

- **Conjunto PU: 1050 composiciones.** Prevalencia de positivas etiquetadas:
  **21.5%** (23.3% con las ambiguas como P).
  No es la prior π, que se estima en F3.2.
- Entradas con veredicto manual (F1.4): 59.

## 3. Banderas (análisis de sensibilidad; no cambian la etiqueta)

| Bandera | Total | Por etiqueta | Composiciones |
|---|---|---|---|
| `flag_high_pressure` | 4 | positive: 3, ambiguous: 1 | CoBiO3, MgSiO3, ScBiO3, VPbO3 |
| `flag_metastable_dft` | 13 | positive: 13 | BaIrO3, CaGeO3, CaSiO3, CdGeO3, GePbO3, LuMnO3, MgSiO3, NaSbO3, SrGeO3, SrSiO3, TiMnO3, VCdO3 … |
| `flag_ambiguous_oxidation` | 18 | positive: 17, ambiguous: 1 | BaBiO3, BaFeO3, BaNpO3, BaOsO3, BaPaO3, BaPuO3, CaFeO3, CaTcO3, CdTcO3, NaOsO3, NiBiO3, SrBiO3 … |
| `flag_bartel_conflict` | 2 | positive: 2 | SrSiO3, TiMnO3 |
| `flag_only_hexagonal` | 8 | synthesized_non_perovskite: 8 | BaCoO3, BaCrO3, BaMnO3, BaNiO3, BaRhO3, BaRuO3, BaTcO3, KTiO3 |
| `flag_icsd_without_exp` | 3 | unlabeled: 3 | NiPbO3, SrRhO3, TiSnO3 |

- `flag_high_pressure`: lista curada en `configs/data.yaml` (ampliable).
- `flag_metastable_dft`: la perovskita experimental queda ≥ 0,05 eV/átomo por encima del
  polimorfo más estable, que no es perovskita (E_hull GGA/GGA+U).
- La sensibilidad de F3.6 excluye **31 positivas**: alta presión, metaestables u
  oxidación ambigua.
