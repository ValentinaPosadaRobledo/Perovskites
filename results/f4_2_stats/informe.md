# F4.2: pruebas confirmatorias de H1 y H2 (validación cruzada)

Entradas: `results/f3_3_pu/20261119T211824Z_aacd15c3f6dd` (modelo `pu_pre_dft`) y `results/f2_4_baselines/20261119T211726Z_131d75672dea` (τ); semillas 0–9; bootstrap jerárquico B = 10000, semilla 20261113; commit `ab07040`.

## H1 — caída de TPR de `random` a cada esquema (modelo PU `pre_dft`)

Criterio: p bilateral con Holm < 0.05 y ΔTPR ≥ δ₁ = 0.05.

| Esquema | ΔTPR | IC 95 % | DE entre semillas | p | p (Holm) | Unidades / P | Cumple |
|---|---|---|---|---|---|---|---|
| `loeo_A` | 0.044 | [-0.003; 0.102] | 0.022 | 0.0712 | 0.0824 | 31 / 193 | no |
| `loeo_B` | 0.049 | [0.002; 0.101] | 0.024 | 0.0412 | 0.0824 | 33 / 193 | no |
| `family` | 0.206 | [0.139; 0.273] | 0.043 | < 1e-04 | < 1e-04 | 33 / 193 | sí |
| `cluster` | 0.125 | [0.074; 0.173] | 0.046 | < 1e-04 | < 1e-04 | 33 / 193 | sí |

**Conclusión global (intersección-unión):** H1 no se sostiene en los 4 esquemas (2 de 4 cumplen el criterio).

## H2 — no inferioridad del PU frente a τ: principal (incluye Bartel, τ *in-sample*)

Criterio: P*(Δ* ≤ −δ₂) con Holm < 0.025 (δ₂ = 0.05).

| Esquema | TPR τ | TPR PU (igual P(ŷ=1)) | Δ | IC 95 % | p no inf. (Holm) | No inferior | Superior | P |
|---|---|---|---|---|---|---|---|---|
| `loeo_A` | 0.922 | 0.940 | +0.018 | [-0.019; +0.056] | 0.0008 | sí | no | 193 |
| `loeo_B` | 0.922 | 0.917 | -0.005 | [-0.049; +0.045] | 0.0708 | no | no | 193 |
| `family` | 0.922 | 0.866 | -0.056 | [-0.135; +0.031] | 0.8178 | no | no | 193 |
| `cluster` | 0.922 | 0.880 | -0.042 | [-0.095; +0.014] | 0.8178 | no | no | 193 |

**Conclusión global:** no inferioridad no demostrada en los 4 esquemas (1 de 4). Un resultado no concluyente no se interpreta como inferioridad ni como equivalencia (§8.4).

## H2 — no inferioridad del PU frente a τ: sensibilidad sin los compuestos de Bartel (τ fuera de muestra)

Criterio: P*(Δ* ≤ −δ₂) con Holm < 0.025 (δ₂ = 0.05).

| Esquema | TPR τ | TPR PU (igual P(ŷ=1)) | Δ | IC 95 % | p no inf. (Holm) | No inferior | Superior | P |
|---|---|---|---|---|---|---|---|---|
| `loeo_A` | 0.878 | 0.902 | +0.024 | [+0.000; +0.102] | < 1e-04 | sí | no | 41 |
| `loeo_B` | 0.878 | 0.856 | -0.022 | [-0.076; +0.025] | 0.3768 | no | no | 41 |
| `family` | 0.878 | 0.700 | -0.178 | [-0.336; -0.003] | 1.0000 | no | no | 41 |
| `cluster` | 0.878 | 0.760 | -0.118 | [-0.219; +0.003] | 1.0000 | no | no | 41 |

**Conclusión global:** no inferioridad no demostrada en los 4 esquemas (1 de 4). Un resultado no concluyente no se interpreta como inferioridad ni como equivalencia (§8.4).

## H1 descriptivo (otros modelos, sin decisión)

| Modelo | Esquema | ΔTPR | DE entre semillas |
|---|---|---|---|
| `pu_post_dft` | `loeo_A` | 0.008 | 0.017 |
| `pu_post_dft` | `loeo_B` | 0.016 | 0.014 |
| `pu_post_dft` | `family` | 0.053 | 0.028 |
| `pu_post_dft` | `cluster` | 0.113 | 0.110 |

## Comparaciones secundarias (Nadeau–Bengio, no confirmatorias)

PU `pu_pre_dft` frente a cada línea base: TPR a P(ŷ=1) = 0.1 en cada fold, sobre las composiciones puntuadas por los dos modelos (empates repartidos, §7.2); t corregido de Nadeau y Bengio con J = semillas × folds con positivos; Holm por esquema (α = 0.05). Las TPR son medias por fold (no micro-agregadas).

| Esquema | Línea base | TPR PU | TPR base | Δ | IC 95 % | J | n_test / n_train | p | p (Holm) |
|---|---|---|---|---|---|---|---|---|---|
| `random` | `pu_post_dft` | 0.368 | 0.383 | -0.015 | [-0.036; +0.006] | 50 | 156 / 624 | 0.1623 | 0.1623 |
| `random` | `ehull_model` | 0.368 | 0.301 | +0.067 | [+0.024; +0.110] | 50 | 156 / 624 | 0.0029 | 0.0058 |
| `random` | `symmetry_model` | 0.368 | 0.296 | +0.071 | [+0.034; +0.108] | 50 | 156 / 624 | 0.0003 | 0.0009 |
| `random` | `tau_bartel` | 0.301 | 0.182 | +0.120 | [+0.065; +0.174] | 50 | 126 / 506 | 5.5e-05 | 0.0002 |
| `random` | `t_window_bartel` | 0.303 | 0.190 | +0.113 | [+0.064; +0.162] | 50 | 127 / 510 | 2.6e-05 | 0.0001 |
| `random` | `charge_balance_icsd` | 0.431 | 0.143 | +0.288 | [+0.245; +0.331] | 50 | 185 / 739 | 5.0e-18 | 3.0e-17 |
| `cluster` | `pu_post_dft` | 0.401 | 0.427 | -0.026 | [-0.143; +0.090] | 41 | 190 / 590 | 0.6511 | 1.0000 |
| `cluster` | `ehull_model` | 0.401 | 0.448 | -0.047 | [-0.167; +0.073] | 41 | 190 / 590 | 0.4332 | 1.0000 |
| `cluster` | `symmetry_model` | 0.401 | 0.383 | +0.018 | [-0.126; +0.162] | 41 | 190 / 590 | 0.8012 | 1.0000 |
| `cluster` | `tau_bartel` | 0.287 | 0.198 | +0.089 | [+0.005; +0.172] | 41 | 154 / 478 | 0.0379 | 0.1893 |
| `cluster` | `t_window_bartel` | 0.288 | 0.226 | +0.062 | [-0.013; +0.136] | 41 | 154 / 483 | 0.1040 | 0.4159 |
| `cluster` | `charge_balance_icsd` | 0.438 | 0.194 | +0.244 | [+0.065; +0.422] | 41 | 224 / 700 | 0.0086 | 0.0516 |
