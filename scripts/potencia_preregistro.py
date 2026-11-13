"""Potencia aproximada de H1 y H2 para fijar δ₁ y δ₂ en el preregistro (F0.4).

Se ejecuta ANTES de tener datos, así que todo depende de supuestos explícitos (abajo).
Sirve para saber si los márgenes elegidos son alcanzables con el número de positivos
esperado (200–350 composiciones ABO3), no para estimar el resultado.

Modelo (aproximación normal para una diferencia pareada de aciertos binarios):
    - Cada positivo tiene dos resultados binarios (acierto / fallo): en H1, el mismo
      modelo bajo `random` y bajo un esquema por grupos; en H2, el modelo PU y τ.
    - d  = tasa de discordancia: fracción de positivos en que los dos resultados difieren.
    - Var(Δ̂) ≈ (d − Δ²) / N · DE, con efecto de diseño DE = 1 + (m − 1)·ρ, donde m = N/G
      es el tamaño medio de grupo (elementos B) y ρ la correlación intragrupo.
    - Se ignora que promediar sobre 10 semillas reduce algo la varianza (conservador).

Reglas de decisión (las del preregistro):
    H1: Δ̂ ≥ δ₁  y  IC 95 % excluye 0    →  potencia = Φ((Δ − max(δ₁, z·SE)) / SE)
    H2: cota inferior del IC 95 % > −δ₂  →  potencia = Φ((Δ + δ₂ − z·SE) / SE)

Uso:
    uv run python scripts/potencia_preregistro.py
"""

from __future__ import annotations

import math
from itertools import product

from scipy.stats import norm

Z = norm.ppf(0.975)  # IC 95 % bilateral
N_GROUPS = 30  # supuesto: ~30 elementos B con positivos (grupos de loeo_B)


def standard_error(n_pos: int, discordance: float, delta: float, icc: float) -> float:
    """Error estándar de la diferencia pareada de TPR con efecto de diseño por grupos."""
    design_effect = 1 + (n_pos / N_GROUPS - 1) * icc
    return math.sqrt(max(discordance - delta**2, 1e-12) / n_pos * design_effect)


def power_h1(n_pos: int, discordance: float, delta: float, icc: float, delta1: float) -> float:
    """P(Δ̂ ≥ δ₁ y Δ̂ − z·SE > 0) cuando la caída real es `delta`."""
    se = standard_error(n_pos, discordance, delta, icc)
    return float(norm.cdf((delta - max(delta1, Z * se)) / se))


def power_h2(n_pos: int, discordance: float, delta: float, icc: float, delta2: float) -> float:
    """P(Δ̂ − z·SE > −δ₂) cuando la diferencia real PU − τ es `delta`."""
    se = standard_error(n_pos, discordance, delta, icc)
    return float(norm.cdf((delta + delta2 - Z * se) / se))


def main() -> None:
    n_values = (150, 250, 350)

    # --- H2: no inferioridad frente a τ -------------------------------------
    print("### H2: probabilidad de declarar no inferioridad\n")
    print("| d | ρ | Δ real (PU − τ) | δ₂ | " + " | ".join(f"N = {n}" for n in n_values) + " |")
    print("|---|---|---|---|" + "---|" * len(n_values))
    for d, icc, delta, delta2 in product((0.10, 0.20), (0.0, 0.05), (0.0, 0.03), (0.03, 0.05)):
        cells = " | ".join(f"{power_h2(n, d, delta, icc, delta2):.2f}" for n in n_values)
        print(f"| {d:.2f} | {icc:.2f} | {delta:+.2f} | {delta2:.2f} | {cells} |")

    # --- H1: caída al pasar de random a grupos -------------------------------
    print("\n### H1: probabilidad de declarar una caída relevante (δ₁ = 0,05)\n")
    print("| d | ρ | Caída real | " + " | ".join(f"N = {n}" for n in n_values) + " |")
    print("|---|---|---|" + "---|" * len(n_values))
    for d, icc, delta in product((0.20, 0.30), (0.0, 0.05), (0.05, 0.10, 0.15)):
        cells = " | ".join(f"{power_h1(n, d, delta, icc, 0.05):.2f}" for n in n_values)
        print(f"| {d:.2f} | {icc:.2f} | {delta:.2f} | {cells} |")


if __name__ == "__main__":
    main()
