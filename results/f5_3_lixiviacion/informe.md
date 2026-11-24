# F5.3: lixiviación de los sitios A y B (tesis, exploratorio)

Entradas: `results/f5_1_pourbaix/20261120T200254Z_069e8fc918f9/pourbaix.csv`, `results/f5_2_pareto/20261124T152358Z_9cfd42873380/candidates.csv`; commit `c60ba27`. Lectura termodinámica del diagrama de Pourbaix: un sitio se lixivia si su metal aparece en alguna especie disuelta estable en la ventana de −0,2 a +0,4 V. No es una cinética de disolución.

## Patrones a pH 7,4 y 1e-6 M

| Grupo | n | Solo A | Solo B | Ambos | Ninguno |
|---|---|---|---|---|---|
| P | 192 | 55% | 2% | 35% | 9% |
| U | 539 | 38% | 8% | 44% | 10% |
| U_eligible_F5.2 | 390 | 41% | 6% | 41% | 12% |
| front_F5.2 | 5 | 60% | 0% | 20% | 20% |

## Frente principal de F5.2 a pH 7,4 y 1e-6 M

| Composición | ΔG_pbx | Patrón | Especies de A | Especies de B | Sólidos |
|---|---|---|---|---|---|
| CaHfO3 | 0.198 | only_A | Ca[+2] | — | HfO2(s) |
| ErAlO3 | 0.041 | only_A | Er[+3] | — | Er3Al5O12(s) |
| EuVO3 | 0.210 | both | Eu[+3] | VO3[-1] | V3O5(s); VO2(s) |
| LuAlO3 | 0.021 | none | — | — | Lu2O3(s); Lu3Al5O12(s) |
| TbGaO3 | 0.069 | only_A | Tb[+3] | — | Tb3Ga5O12(s) |
