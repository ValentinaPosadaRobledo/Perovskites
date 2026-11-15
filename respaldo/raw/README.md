# Respaldo de los `raw/` por composición (2026-11-15)

`results/**/raw/` no se versiona (`.gitignore`), pero F3.6 (informe pareado), F4.2 (pruebas
por composición) y los informes de F3.3/F3.4 los necesitan. Aquí están, comprimidos con
`gzip -n -9`, los CSV de las cuatro corridas completadas (sin `parts/`, que duplica su
contenido). Solo composiciones, etiquetas y puntuaciones; nada de ICSD.

| Experimento | Corrida | Commit de la corrida |
|---|---|---|
| F2.4 `f2_4_baselines` | `20261114T013022Z_00a61daabfdf` | `922443b` |
| F3.2 `f3_2_prior` | `20261115T170049Z_19f79970b85a` | `60fabb5` |
| F3.3 `f3_3_pu` | `20261114T164434Z_aacd15c3f6dd` | `3c24acf` |
| F3.4 `f3_4_formability` | `20261115T170454Z_b6b371e72440` | `d662b65` |

## Restaurar (desde la raíz del repositorio, en `main`)

```bash
git fetch origin respaldo-raw
git archive origin/respaldo-raw respaldo/raw | tar -x
for f in $(find respaldo/raw/results -name '*.csv.gz'); do d=${f#respaldo/raw/}; mkdir -p "$(dirname "$d")"; gunzip -c "$f" > "${d%.gz}"; done
sha256sum -c respaldo/raw/SHA256SUMS
rm -r respaldo
```

`SHA256SUMS` tiene el hash de cada CSV ORIGINAL (sin comprimir), con su ruta en `results/`:
`sha256sum -c` debe dar `OK` en los 65.
