# perovskite-pu

**Evaluación de la generalización de modelos de aprendizaje positivo–no etiquetado (PU
learning) para predecir la sintetizabilidad de perovskitas óxido ABO₃ con control de fuga
de información.**

Tesis de maestría y artículo científico (revista objetivo: *Digital Discovery*). El núcleo
es un benchmark preregistrado de sintetizabilidad PU bajo extrapolación química
(particiones por elemento, familia y clúster), con reevaluación del modelo de grafos de
Gu et al. (2022).

- Reglas permanentes del proyecto: [`CLAUDE.md`](CLAUDE.md)
- Plan, estado y bitácora de decisiones: [`ROADMAP.md`](ROADMAP.md)
- Preregistro (borrador): [`docs/preregistro.md`](docs/preregistro.md)

## Estado

Fase 0 (inicialización). Aún no hay datos descargados ni modelos entrenados.

## Instalación

Requiere [uv](https://docs.astral.sh/uv/). uv instala Python 3.12 si hace falta.

```bash
uv sync
uv run pre-commit install
cp .env.example .env   # y escribir la clave MP_API_KEY de Materials Project
```

## Tareas frecuentes

```bash
uv run poe test        # pytest -rs (tests de fuga y métricas con datos sintéticos)
uv run poe test-data   # igual, pero los tests dependientes de datos FALLAN si faltan datos
uv run poe check       # lint + formato + tests (lo mismo que la CI)
uv run poe figures     # regenera figuras y tablas del artículo desde results/
uv run poe roadmap start F1.1
```

## Estructura

```
configs/          seeds.yaml (única fuente de semillas), data.yaml, experiments/
data/             raw/ (no versionado), interim/, processed/, splits/ (holdout + HASHES.json)
docs/             preregistro, validación manual, prompt de inicialización
src/perovskite_pu/
  data/           descarga, identificación por conectividad, deduplicación, etiquetado
  features/       MAGPIE, t, μ, τ, estados de oxidación
  splits/         random, group_system, loeo_A, loeo_B, family, cluster, temporal
  training/       bagging PU, CV anidada (no puede leer el holdout)
  evaluation/     métricas PU, tests estadísticos, load_holdout()
  pourbaix/       extensión de tesis (estabilidad acuosa)
  utils/          semillas, hashing canónico, metadatos de corrida
tests/            test_leakage.py, test_metrics.py, test_utils.py, fixtures sintéticas
results/          una carpeta por experimento; summary.csv y runs_index.csv
paper/, thesis/   esquemas, scripts de figuras, figuras y tablas
```

## Control de fuga de información (resumen)

- Etiqueta a nivel de composición; deduplicación antes de particionar.
- Holdout externo congelado definido por elementos B reservados; solo se lee con
  `load_holdout()` y `HOLDOUT_UNLOCK=1`, después de depositar el preregistro en OSF.
- `tests/test_leakage.py` verifica en cada ejecución: no solapamiento de composiciones ni
  de grupos, preprocesamiento ajustado solo con train, análisis estático de `training/` y
  un audit hook que impide abrir el holdout durante una corrida.

## Datos y licencias

Las estructuras de ICSD **no se redistribuyen**; solo se publican identificadores y
features derivadas. Los datos crudos de Materials Project se descargan con los scripts del
repositorio, y su versión queda registrada en `data/MANIFEST.json`.

## Uso de IA

El repositorio se desarrolla con asistencia de Claude Code (Anthropic). Su uso se declarará
según la política de la revista; la IA no figura como autora.

## Autoría

Valentina Posada.
