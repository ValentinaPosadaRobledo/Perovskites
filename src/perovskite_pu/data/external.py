"""F1.2: datos externos de Gu et al. (2022) y Bartel et al. (2019), y su cruce con MP.

Uso (desde la raíz del repositorio):

    uv run python -m perovskite_pu.data.external download   # descarga y verifica checksums
    uv run python scripts/report_f1_2.py                     # informe de solapamientos

Detalles verificados al explorar los datos (2026-11-13):

- **Gu et al.** (Zenodo 6348981): `prediction.csv` indexa los cristales con su columna `#`,
  que es el índice que usan las particiones (`splits_Perov_All.json`). Los archivos de
  `data/` (CIF, `id_prop.csv`, `cif_sources.json`) usan OTRO índice, el del archivo CIF. Se
  enlazan por el conjunto de fuentes (IDs de MP/OQMD/AFLOW), que es único por cristal. En
  `prediction.csv`, `ICSD` vale True o está vacío (vacío = virtual).
- **IDs de MP:** MP pasó a identificadores alfabéticos (`mp-aaaaahtd`); los artículos usan
  los numéricos (`mp-5229`). La conversión es determinista (base 26, `emmet.core.mpid.AlphaID`).
- **Bartel et al.** (`TableS1.csv`): 576 ABX3 experimentales; `exp_label` = 1 si es
  perovskita y -1 si no. Los -1 son compuestos SINTETIZADOS no perovskita: sirven para
  validar FORMABILIDAD, no sintetizabilidad (regla 2.3).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import urllib.request
import zipfile
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from emmet.core.mpid import AlphaID
from pymatgen.core import Composition

from perovskite_pu.data.download import MANIFEST_PATH, is_abo3_oxide
from perovskite_pu.utils.hashing import sha256_file
from perovskite_pu.utils.paths import CONFIGS_DIR, RAW_DIR

GU_DIR = RAW_DIR / "gu2022"
GU_ZIP = GU_DIR / "gu2022_release.zip"
GU_EXTRACTED = GU_DIR / "extracted"
BARTEL_DIR = RAW_DIR / "bartel2019" / "repo"
_ELEMENT_RE = re.compile(r"[A-Z][a-z]?")


# ---------------------------------------------------------------------------
# Funciones puras
# ---------------------------------------------------------------------------
def to_legacy_mpid(material_id: str) -> str:
    """Convierte un ID de MP al formato numérico antiguo: `mp-aaaaahtd` -> `mp-5229`.

    Los IDs que ya son numéricos se devuelven igual.
    """
    if re.fullmatch(r"mp-\d+", material_id):
        return material_id
    return f"mp-{int(AlphaID(material_id))}"


def reduced_formula(formula: str) -> str:
    """Fórmula reducida canónica de pymatgen (clave de composición en todo el proyecto)."""
    return Composition(formula).reduced_formula


def gu_root(extracted_dir: Path = GU_EXTRACTED) -> Path:
    """Carpeta raíz del repositorio de Gu dentro del zip extraído (nombre con hash del commit)."""
    roots = [p for p in extracted_dir.iterdir() if p.is_dir()]
    if len(roots) != 1:
        raise FileNotFoundError(f"Se esperaba una carpeta en {extracted_dir}; hay {len(roots)}.")
    return roots[0]


def load_gu2022(root: Path, split_index: int = 0) -> pd.DataFrame:
    """Tabla de cristales de Gu et al., una fila por cristal (índice `#` de prediction.csv).

    Columnas: idx, cif_id, formula (ABC3 tal como la escriben), gu_A, gu_B, gu_X,
    reduced_formula, is_positive (cristal perovskita con ICSD), cl_score, score_std,
    sources, mp_legacy_ids, abo3_oxide y split (PTrain/PVal/PTest/NTrain/NVal/NTest).
    """
    pred = pd.read_csv(root / "prediction.csv").rename(columns={"#": "idx"})
    pred["is_positive"] = pred["ICSD"].notna() & pred["ICSD"].astype(str).eq("True")
    # Enlace con el índice de archivo CIF a través del conjunto de fuentes (único por cristal).
    cif_sources = json.loads((root / "data" / "cif_sources.json").read_text())
    key_to_cif = {frozenset(map(str, ids)): cid for cid, ids in cif_sources.items()}
    if len(key_to_cif) != len(cif_sources):
        raise ValueError("Hay conjuntos de fuentes repetidos: el enlace cristal-CIF sería ambiguo.")
    pred["sources"] = pred["sources"].astype(str).str.split(",")
    pred["cif_id"] = [key_to_cif.get(frozenset(s)) for s in pred["sources"]]
    if pred["cif_id"].isna().any():
        raise ValueError(f"{pred['cif_id'].isna().sum()} cristales sin CIF asociado.")

    elements = pred["ABC3"].map(_ELEMENT_RE.findall)
    pred["gu_A"], pred["gu_B"], pred["gu_X"] = zip(*elements, strict=True)
    pred["reduced_formula"] = pred["ABC3"].map(reduced_formula)
    pred["abo3_oxide"] = pred["ABC3"].map(is_abo3_oxide)
    pred["mp_legacy_ids"] = pred["sources"].map(lambda s: [x for x in s if x.startswith("mp-")])

    split = json.loads((root / "splits_Perov_All.json").read_text())[split_index]
    membership = {i: part for part, idxs in split.items() for i in idxs}
    pred["split"] = pred["idx"].map(membership)
    columns = {"ABC3": "formula", "CL score": "cl_score", "score std": "score_std"}
    keep = [
        "idx", "cif_id", "formula", "gu_A", "gu_B", "gu_X", "reduced_formula", "is_positive",
        "cl_score", "score_std", "sources", "mp_legacy_ids", "abo3_oxide", "split",
    ]  # fmt: skip
    return pred.rename(columns=columns)[keep]


def gu_test_composition_leakage(gu: pd.DataFrame, only_abo3: bool = False) -> dict[str, Any]:
    """¿Cuántos positivos de test de Gu tienen su composición también en train/val?

    Es el dato de partida de S2 (preregistro §10b): Gu particiona por cristal, así que
    polimorfos de una misma composición pueden quedar a ambos lados.
    """
    df = gu[gu["abo3_oxide"]] if only_abo3 else gu
    test = df[df["split"] == "PTest"]
    seen = set(gu.loc[gu["split"].isin(["PTrain", "PVal", "NTrain", "NVal"]), "reduced_formula"])
    seen_pos = set(gu.loc[gu["split"].isin(["PTrain", "PVal"]), "reduced_formula"])
    n = len(test)
    shared = int(test["reduced_formula"].isin(seen).sum())
    shared_pos = int(test["reduced_formula"].isin(seen_pos).sum())
    return {
        "test_positives": n,
        "composition_in_train_or_val": shared,
        "composition_in_train_or_val_frac": shared / n if n else float("nan"),
        "composition_among_train_val_positives": shared_pos,
        "composition_among_train_val_positives_frac": shared_pos / n if n else float("nan"),
    }


def load_bartel2019(table: Path) -> pd.DataFrame:
    """TableS1 de Bartel et al. con `reduced_formula`, `abo3_oxide` y `is_perovskite`."""
    t = pd.read_csv(table)
    t["reduced_formula"] = t["ABX3"].map(reduced_formula)
    t["abo3_oxide"] = t["ABX3"].map(is_abo3_oxide)
    t["is_perovskite"] = t["exp_label"] == 1
    return t


def composition_labels(df: pd.DataFrame, flag: str, key: str = "reduced_formula") -> pd.Series:
    """Etiqueta por composición: True si ALGUNA fila de esa composición tiene `flag` True."""
    return df.groupby(key)[flag].any()


def label_conflicts(bartel: pd.DataFrame, gu: pd.DataFrame) -> pd.DataFrame:
    """Óxidos ABO3 cuya etiqueta difiere entre Bartel (perovskita sí/no) y Gu (positivo sí/no).

    Tipos: `bartel_no_gu_si` (sintetizado NO perovskita para Bartel, positivo para Gu) y
    `bartel_si_gu_virtual` (perovskita para Bartel, solo virtual en Gu).
    """
    b = bartel[bartel["abo3_oxide"]].set_index("reduced_formula")["is_perovskite"]
    g = composition_labels(gu[gu["abo3_oxide"]], "is_positive")
    both = b.index.intersection(g.index)
    rows = []
    for comp in sorted(both):
        if not b[comp] and g[comp]:
            rows.append({"reduced_formula": comp, "type": "bartel_no_gu_si"})
        elif b[comp] and not g[comp]:
            rows.append({"reduced_formula": comp, "type": "bartel_si_gu_virtual"})
    return pd.DataFrame(rows, columns=["reduced_formula", "type"])


# ---------------------------------------------------------------------------
# Descarga y registro en el manifiesto
# ---------------------------------------------------------------------------
def _md5(path: Path) -> str:
    h = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def safe_extract(zip_path: Path, out_dir: Path, extensions: Iterable[str]) -> int:
    """Extrae solo las extensiones indicadas, rechazando rutas que escapen de `out_dir`."""
    out_dir = out_dir.resolve()
    exts = tuple(extensions)
    n = 0
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if info.is_dir() or not info.filename.endswith(exts):
                continue
            dest = (out_dir / info.filename).resolve()
            if not dest.is_relative_to(out_dir):
                raise ValueError(f"Ruta peligrosa en el zip: {info.filename}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(z.read(info))
            n += 1
    return n


def download_gu2022(cfg: dict[str, Any]) -> dict[str, Any]:
    """Descarga el zip de Zenodo, verifica su MD5 publicado y extrae los tipos seguros."""
    GU_DIR.mkdir(parents=True, exist_ok=True)
    if not GU_ZIP.exists() or _md5(GU_ZIP) != cfg["md5"]:
        url = f"https://zenodo.org/api/records/{cfg['zenodo_record']}/files/{cfg['file']}/content"
        with urllib.request.urlopen(url, timeout=600) as r:
            GU_ZIP.write_bytes(r.read())
    md5 = _md5(GU_ZIP)
    if md5 != cfg["md5"]:
        raise ValueError(f"MD5 del zip de Gu distinto del publicado: {md5} != {cfg['md5']}")
    n = safe_extract(GU_ZIP, GU_EXTRACTED, cfg["extract_extensions"])
    return {"zenodo_record": cfg["zenodo_record"], "md5": md5, "sha256": sha256_file(GU_ZIP),
            "extracted_files": n}  # fmt: skip


def download_bartel2019(cfg: dict[str, Any]) -> dict[str, Any]:
    """Obtiene el repositorio de Bartel en el commit fijado y registra el hash de la tabla."""
    if not (BARTEL_DIR / ".git").exists():
        BARTEL_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", "-q"], cwd=BARTEL_DIR, check=True)
        subprocess.run(["git", "remote", "add", "origin", cfg["repo"]], cwd=BARTEL_DIR, check=True)
    subprocess.run(
        ["git", "fetch", "-q", "--depth", "1", "origin", cfg["commit"]], cwd=BARTEL_DIR, check=True
    )
    subprocess.run(["git", "checkout", "-q", cfg["commit"]], cwd=BARTEL_DIR, check=True)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=BARTEL_DIR, capture_output=True, text=True, check=True
    ).stdout.strip()
    if head != cfg["commit"]:
        raise ValueError(f"Commit de Bartel distinto del fijado: {head} != {cfg['commit']}")
    return {"repo": cfg["repo"], "commit": head, "table": cfg["table"],
            "table_sha256": sha256_file(BARTEL_DIR / cfg["table"])}  # fmt: skip


def register_external(manifest_path: Path, records: dict[str, Any]) -> dict[str, Any]:
    """Añade la sección `external` al manifiesto sin tocar el registro de MP."""
    manifest = json.loads(manifest_path.read_text("utf-8")) if manifest_path.exists() else {}
    manifest["external"] = {
        **records,
        "retrieved_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="F1.2: datos de Gu et al. y Bartel et al.")
    parser.add_argument("command", choices=["download"])
    parser.add_argument("--config", type=Path, default=CONFIGS_DIR / "data.yaml")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text("utf-8"))["external_sources"]
    records = {
        "gu2022": download_gu2022(cfg["gu2022"]),
        "bartel2019": download_bartel2019(cfg["bartel2019"]),
    }
    register_external(MANIFEST_PATH, records)
    print(json.dumps(records, indent=2))


if __name__ == "__main__":  # requisito en Windows (regla 2.2)
    main()
