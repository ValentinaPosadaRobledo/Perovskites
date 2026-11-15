#!/usr/bin/env bash
# F3.5, S2: entrena las variantes (a) por cristal (sus particiones originales) y (b) agrupada por
# composición (mismo test) para las particiones 0..N-1. Reanudable: cada modelo terminado deja
# su `test_xxx.json` y no se repite. Semilla: la primera de configs/seeds.yaml del proyecto.
# Uso: scripts/run_s2.sh <N> [hilos]
set -euo pipefail
G=/home/user/gu_isolated
REPO=$G/repo/kaist-amsg-PerovskiteSynthesizability_Manuscript2021-e12089a
N=${1:?número de particiones}
THREADS=${2:-4}
SEED=$(python3 -I -c "import re; t=open('/home/user/perovskites/configs/seeds.yaml').read(); print(re.search(r'^seeds:\s*\[(\d+)', t, re.M).group(1))")
cd "$G"
for i in $(seq 0 $((N - 1))); do
  for v in a b; do
    if [ "$v" = a ]; then SPL=$REPO/splits_Perov_All.json; else SPL=$G/work/splits_S2_grouped.json; fi
    echo "[$(date -u +%H:%M:%S)] variante $v, partición $i"
    OMP_NUM_THREADS=$THREADS .venv/bin/python -I scripts/train_split.py --repo "$REPO" --work "$G/work" \
      --shim "$G/shim" --splits "$SPL" --split "$i" --variant "$v" --seed "$SEED" --threads "$THREADS" \
      > /dev/null 2>> "$G/work/s2_errors.log"
  done
done
echo "[$(date -u +%H:%M:%S)] S2 terminado ($N particiones)"
