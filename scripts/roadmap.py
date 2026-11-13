"""Actualiza el estado de una tarea en ROADMAP.md (regla 2.5 de CLAUDE.md).

Uso:
    uv run poe roadmap start F1.1
    uv run poe roadmap done F1.1 --result "results/f1_1/summary.csv"
    uv run poe roadmap done F1.1 --commit abc1234
    uv run poe roadmap block F1.1 --note "Esperando la clave de MP"

`done` usa por defecto `git rev-parse --short HEAD`: ejecútalo justo DESPUÉS del commit
que produjo el resultado y haz la actualización del ROADMAP en un commit aparte.
Columnas esperadas: ID | Tarea | Destino | Estado | Inicio | Fin | Commit | Resultado / Notas
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date
from pathlib import Path

ROADMAP = Path(__file__).resolve().parents[1] / "ROADMAP.md"

# Posición de cada columna en la fila (0 y último son vacíos por los "|" de los extremos).
COL = {"id": 1, "estado": 4, "inicio": 5, "fin": 6, "commit": 7, "notas": 8}
STATUS = {"pending": "⬜ Pendiente", "start": "🔄 En proceso", "done": "✅ Completada"}


def short_head() -> str:
    """Hash corto del commit actual."""
    out = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


def update_row(line: str, args: argparse.Namespace, today: str) -> str:
    """Devuelve la fila de la tabla con las columnas actualizadas según el comando."""
    cells = line.split("|")
    if len(cells) < COL["notas"] + 2:
        sys.exit(f"Fila con formato inesperado: {line!r}")

    def set_cell(name: str, value: str) -> None:
        cells[COL[name]] = f" {value} "

    def get_cell(name: str) -> str:
        return cells[COL[name]].strip()

    if args.command == "start":
        set_cell("estado", STATUS["start"])
        if not get_cell("inicio"):
            set_cell("inicio", today)
    elif args.command == "done":
        set_cell("estado", STATUS["done"])
        if not get_cell("inicio"):
            set_cell("inicio", today)
        set_cell("fin", today)
        set_cell("commit", f"`{args.commit or short_head()}`")
        if args.result:
            set_cell("notas", args.result)
    elif args.command == "block":
        # Una tarea bloqueada sigue "En proceso" con el motivo en notas.
        set_cell("estado", STATUS["start"])
        previous = get_cell("notas")
        note = f"**Bloqueada ({today}):** {args.note}"
        set_cell("notas", f"{previous} · {note}" if previous else note)
    return "|".join(cells)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p_start = sub.add_parser("start", help="Marca la tarea como 🔄 En proceso")
    p_start.add_argument("task_id")
    p_done = sub.add_parser("done", help="Marca la tarea como ✅ Completada")
    p_done.add_argument("task_id")
    p_done.add_argument("--commit", help="Commit que produjo el resultado (por defecto HEAD)")
    p_done.add_argument("--result", help="Enlace o descripción del resultado")
    p_block = sub.add_parser("block", help="Deja la tarea en 🔄 con el motivo del bloqueo")
    p_block.add_argument("task_id")
    p_block.add_argument("--note", required=True)
    p_block.add_argument("--date", help=argparse.SUPPRESS)
    for p in (p_start, p_done):
        p.add_argument("--date", help=argparse.SUPPRESS)  # para tests reproducibles
    args = parser.parse_args(argv)

    today = args.date or date.today().isoformat()
    lines = ROADMAP.read_text(encoding="utf-8").splitlines(keepends=True)
    prefix = f"| {args.task_id} |"
    matches = [i for i, ln in enumerate(lines) if ln.startswith(prefix)]
    if len(matches) != 1:
        sys.exit(f"Se esperaba exactamente una fila para {args.task_id}; hay {len(matches)}.")
    i = matches[0]
    newline = "\n" if lines[i].endswith("\n") else ""
    lines[i] = update_row(lines[i].rstrip("\n"), args, today) + newline
    # newline="\n": LF también en Windows.
    with open(ROADMAP, "w", encoding="utf-8", newline="\n") as fh:
        fh.writelines(lines)
    print(f"{args.task_id}: {args.command} ({today})")


if __name__ == "__main__":
    main()
