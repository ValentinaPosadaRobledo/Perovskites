# F4.2: pruebas confirmatorias de H1 y H2 (validación cruzada)

Entradas: `results/f3_3_pu/20261119T211824Z_aacd15c3f6dd` (modelo `pu_pre_dft`) y `results/f2_4_baselines/20261119T211726Z_131d75672dea` (τ); semillas 0–9; bootstrap jerárquico B = 10000, semilla 20261113; commit `8fa9fd3`.

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
