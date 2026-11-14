"""Factores geométricos: Goldschmidt t, factor octaédrico μ y τ de Bartel et al. (2019).

Reimplementación fiel del algoritmo de `PredictPerovskites.py` de Bartel et al. (repositorio
CJBartel/perovskite-stability, commit 965b8de, licencia MIT), validada reproduciendo los τ de
su TableS1 (tests/test_tolerance.py y `scripts/report_f2_1.py`).

Algoritmo (para ABX3):
1. Estados de oxidación: los cationes típicamente 1+ (H, Li, Na, K, Rb, Cs, Fr, Ag) o 2+
   (Be, Mg, Ca, Sr, Ba, Ra) quedan fijos; el resto puede tomar los estados que tienen radio de
   Shannon. Se conservan las combinaciones que equilibran X3. Si hay varias, se desempata:
   - dos combinaciones que son una la inversa de la otra: el catión más electropositivo lleva
     la carga mayor;
   - si un catión es típicamente 3+ (Sc, Y, La, Al, Ga, In, lantánidos), se toma (3, 3);
   - si las electronegatividades son parecidas (mín > 0,9·máx), se minimiza |n1 − n2|; si no,
     se maximiza. Los empates restantes se resuelven por electronegatividad.
2. Radios de Shannon: para el sitio A, el del NC disponible más cercano a 12; para B, el más
   cercano a 6 (se registra el NC usado: si no es 12 o 6, el radio es un SUSTITUTO).
3. A es el catión más grande según los criterios de Bartel; τ no está definido si r_A ≤ r_B.

Fórmulas: t = (r_A + r_X) / (√2 (r_B + r_X)); μ = r_B / r_X;
τ = r_X/r_B − n_A (n_A − (r_A/r_B)/ln(r_A/r_B)). Perovskita si τ < 4,18.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path

ShannonDict = dict[str, dict[int, dict[int, float]]]

X_OX = {"N": -3, "O": -2, "S": -2, "Se": -2, "F": -1, "Cl": -1, "Br": -1, "I": -1}
PLUS_ONE = {"H", "Li", "Na", "K", "Rb", "Cs", "Fr", "Ag"}
PLUS_TWO = {"Be", "Mg", "Ca", "Sr", "Ba", "Ra"}
PLUS_THREE = {"Sc", "Y", "La", "Al", "Ga", "In", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
              "Dy", "Ho", "Er", "Tm", "Yb", "Lu"}  # fmt: skip
TAU_THRESHOLD = 4.18


@dataclass
class TauResult:
    """Asignación y descriptores de un compuesto ABX3 según Bartel et al. (2019)."""

    A: str | None = None
    B: str | None = None
    X: str | None = None
    nA: int | None = None
    nB: int | None = None
    rA: float | None = None
    rB: float | None = None
    rX: float | None = None
    A_cn: int | None = None  # NC de Shannon usado para r_A (12 = nominal)
    B_cn: int | None = None  # NC de Shannon usado para r_B (6 = nominal)
    t: float = math.nan
    mu: float = math.nan
    tau: float = math.nan
    notes: list[str] = field(default_factory=list)

    def as_row(self) -> dict:
        row = asdict(self)
        row["notes"] = ";".join(self.notes)
        return row


def load_shannon(path: Path | str) -> ShannonDict:
    """Radios de Shannon de Bartel: {elemento: {oxidación: {NC: radio}}}, conservando el orden."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {el: {int(ox): {int(cn): v["only_spin"] for cn, v in cns.items()}
                 for ox, cns in oxs.items()} for el, oxs in raw.items()}  # fmt: skip


def load_electronegativity(path: Path | str) -> dict[str, float]:
    """Electronegatividades de Pauling del repositorio de Bartel."""
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            el, val = line.split(",")[:2]
            out[el.strip()] = float(val)
    return out


def default_tables() -> tuple[ShannonDict, dict[str, float]]:
    """Radios de Shannon y electronegatividades de Bartel incluidos en el paquete.

    Son copias sin modificar de `Shannon_radii_dict.json` y `electronegativities.csv` del
    repositorio de Bartel (licencia MIT, aviso en `features/data/LICENSE_bartel2019.txt`).
    Se distribuyen con el paquete para que τ no dependa de una descarga.
    """
    from importlib.resources import files

    base = files("perovskite_pu.features") / "data"
    return (
        load_shannon(str(base / "Shannon_radii_dict.json")),
        load_electronegativity(str(base / "electronegativities.csv")),
    )


def _parse(formula: str) -> tuple[list[str], str]:
    """Cationes (en el orden de la fórmula) y anión X (el elemento con subíndice 3)."""
    pairs = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    xs = [el for el, n in pairs if n == "3"]
    if len(pairs) != 3 or len(xs) != 1:
        raise ValueError(f"No es ABX3: {formula}")
    return [el for el, _ in pairs if el != xs[0]], xs[0]


def _choose_ox(cations: list[str], X: str, combos: list[tuple[int, int]],
               chi: Mapping[str, float]) -> dict[str, int] | None:  # fmt: skip
    """Elección del par de estados de oxidación (lógica de `chosen_ox_states`)."""
    c1, c2 = cations
    epos = min(cations, key=lambda e: chi[e])  # el más electropositivo
    eneg = c2 if epos == c1 else c1

    def by_chi(sub: list[tuple[int, int]]) -> dict[str, int]:
        lo, hi = min(min(c) for c in sub), max(max(c) for c in sub)
        return {epos: hi, eneg: lo}

    if len(combos) == 1:
        return dict(zip(cations, combos[0], strict=True))
    if len(combos) == 2 and combos[0] == combos[1][::-1]:
        return by_chi(combos)
    if c1 in PLUS_THREE or c2 in PLUS_THREE:
        # En el original, si (3,3) no está, la variable queda sin asignar y falla.
        if X_OX[X] == -2 and (3, 3) in combos:
            return {c1: 3, c2: 3}
        return None
    diffs = [abs(a - b) for a, b in combos]
    best = min(diffs) if min(chi.values()) > 0.9 * max(chi.values()) else max(diffs)
    idx = [i for i, d in enumerate(diffs) if d == best]
    if len(idx) == 1:
        return dict(zip(cations, combos[idx[0]], strict=True))
    return by_chi([combos[i] for i in idx])


def _nearest_cn(cns: list[int], target: int) -> int:
    """NC disponible más cercano al objetivo (en empate, el primero en el orden de la tabla)."""
    return min(cns, key=lambda c: abs(c - target))


def tau_descriptors(formula: str, shannon: ShannonDict, chi: Mapping[str, float]) -> TauResult:
    """t, μ y τ de un ABX3 con el algoritmo de Bartel et al. (2019)."""
    res = TauResult()
    try:
        cations, X = _parse(formula)
    except ValueError as exc:
        res.notes.append(str(exc))
        return res
    res.X = X
    if X not in X_OX or any(c not in shannon for c in cations):
        res.notes.append("anión o catión sin radios de Shannon")
        return res
    allowed = {}
    for c in cations:
        allowed[c] = (
            [1] if c in PLUS_ONE else [2] if c in PLUS_TWO else [o for o in shannon[c] if o > 0]
        )
    combos = [
        (a, b) for a in allowed[cations[0]] for b in allowed[cations[1]] if a + b == -3 * X_OX[X]
    ]
    if not combos:
        res.notes.append("sin combinación de oxidación que equilibre la carga")
        return res
    if any(c not in chi for c in cations):
        res.notes.append("sin electronegatividad")
        return res
    ox = _choose_ox(cations, X, combos, {c: chi[c] for c in cations})
    if ox is None:
        res.notes.append("ninguna regla de desempate aplicable (falla en el original)")
        return res

    radii = {}
    for el, o in ox.items():
        cns = list(shannon[el][o])
        a_cn, b_cn = _nearest_cn(cns, 12), _nearest_cn(cns, 6)
        radii[el] = {
            "A": shannon[el][o][a_cn],
            "B": shannon[el][o][b_cn],
            "A_cn": a_cn,
            "B_cn": b_cn,
        }
    e1, e2 = list(radii)
    r1, r2 = radii[e1], radii[e2]
    # Criterios de `pred_A`, en el mismo orden.
    if r1["A"] > r2["B"] and r1["B"] > r2["A"]:
        A = e1
    elif r1["A"] < r2["B"] and r1["B"] < r2["A"]:
        A = e2
    elif r1["A"] > r2["A"] and r1["B"] > r2["B"]:
        A = e1
    elif r1["A"] < r2["A"] and r1["B"] < r2["B"]:
        A = e2
    elif r1["B"] != r2["B"]:
        A = e1 if r1["B"] > r2["B"] else e2
    elif r1["A"] != r2["A"]:
        A = e1 if r1["A"] > r2["A"] else e2
    else:
        A = e1 if ox[e1] < ox[e2] else e2
    B = e2 if e1 == A else e1

    rX = shannon[X][X_OX[X]][4 if X == "N" else 6]
    res.A, res.B, res.nA, res.nB = A, B, ox[A], ox[B]
    res.rA, res.rB, res.rX = radii[A]["A"], radii[B]["B"], rX
    res.A_cn, res.B_cn = radii[A]["A_cn"], radii[B]["B_cn"]
    if res.A_cn != 12:
        res.notes.append(f"r_A sustituto (NC {res.A_cn} en lugar de 12)")
    if res.B_cn != 6:
        res.notes.append(f"r_B sustituto (NC {res.B_cn} en lugar de 6)")
    res.mu = res.rB / rX
    res.t = (res.rA + rX) / (math.sqrt(2) * (res.rB + rX))
    if res.rA <= 0 or res.rB <= 0:
        # Shannon da radios negativos para el H+; el original (np.log) devolvía NaN en silencio.
        res.notes.append("τ no definido: radio de Shannon no positivo")
    elif res.rA <= res.rB:
        res.notes.append("τ no definido: r_A ≤ r_B")
    else:
        q = res.rA / res.rB
        res.tau = rX / res.rB - res.nA * (res.nA - q / math.log(q))
    return res
