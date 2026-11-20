# Respaldo de los `raw/` por composición (2026-11-15; ampliado el 2026-11-20)

`results/**/raw/` no se versiona (`.gitignore`), pero F3.6 (informe pareado), F4.2 (pruebas
por composición) y los informes de F3.3/F3.4 los necesitan. Aquí están, comprimidos con
`gzip -n -9`, los CSV de las corridas completadas (sin `parts/`, que duplica su
contenido). Solo composiciones, etiquetas y puntuaciones; nada de ICSD.

| Experimento | Corrida | Commit de la corrida |
|---|---|---|
| F2.4 `f2_4_baselines` | `20261114T013022Z_00a61daabfdf` | `922443b` |
| F3.2 `f3_2_prior` | `20261115T170049Z_19f79970b85a` | `60fabb5` |
| F3.3 `f3_3_pu` | `20261114T164434Z_aacd15c3f6dd` | `3c24acf` |
| F3.4 `f3_4_formability` | `20261115T170454Z_b6b371e72440` | `d662b65` |
| F2.4 `f2_4_baselines` (reejecución) | `20261119T211726Z_131d75672dea` | `675d445` |
| F3.3 `f3_3_pu` (reejecución) | `20261119T211824Z_aacd15c3f6dd` | `ff8cbf4` |
| F3.6 `f3_6_s01_high_pressure` | `20261115T184333Z_d1028b6ddf1d` | `429f098` |
| F3.6 `f3_6_s02_ambiguous_oxidation` | `20261115T221528Z_49646ae78eac` | `afa8a2c` |
| F3.6 `f3_6_s04_r2scan` | `20261118T120204Z_51570e4c46d7` | `54ade43` |
| F3.6 `f3_6_s05_angle140` | `20261117T190139Z_dbd8a4c70fc6` | `b8a2b4b` |
| F3.6 `f3_6_s06_loeo_B_unmerged` | `20261118T093224Z_711a869c84c2` | `d1100e0` |
| F3.6 `f3_6_s08_recall` | `20261117T145342Z_61120c8a1806` | `3e6911c` |
| F3.6 `f3_6_s11_charge_balanced_u` | `20261116T014025Z_76ce3fe7ee89` | `42647f9` |
| F3.6 `f3_6_s12_sites_by_size` | `20261118T063122Z_9a172377b989` | `a32271c` |
| F3.6 `f3_6_s13_ambiguous_as_positive` | `20261117T224258Z_0b509e3dc698` | `073ad0b` |
| F3.6 `f3_6_s14_pre_dft_tau` | `20261118T024851Z_cd8cbdfaa4ee` | `dacdfc9` |

Las reejecuciones de F2.4 y F3.3 dieron un `summary.csv` idéntico al de la corrida original
(diferencia máxima 0). `scripts/report_f4_2.py` y `scripts/report_f3_6.py` leen las
reejecuciones (commit `c1ea4c5` de `main`).

## Restaurar (desde la raíz del repositorio, en `main`)

```bash
git fetch origin respaldo-raw
git archive origin/respaldo-raw respaldo/raw | tar -x
for f in $(find respaldo/raw/results -name '*.csv.gz'); do d=$(dirname "${f#respaldo/raw/}")/raw; mkdir -p "$d"; gunzip -c "$f" > "$d/$(basename "${f%.gz}")"; done
sha256sum -c respaldo/raw/SHA256SUMS
rm -r respaldo
```

Los `.csv.gz` están en `respaldo/raw/results/<experimento>/<corrida>/` (sin la carpeta
`raw/`, que el bucle reconstruye). `SHA256SUMS` tiene el hash de cada CSV ORIGINAL (sin comprimir), con su ruta en `results/`:
`sha256sum -c` debe dar `OK` en los 312.
Probado el 2026-11-15 (65 OK) y el 2026-11-20 (312 OK, en un directorio vacío).
