#!/usr/bin/env bash
# Evaluate every architecture at every surviving numbered snapshot.
#
# The shared checkpoint rule (DLC snapshot-best by internal test.mAP) depends on
# five validation frames whose labels are suspect, and the HRNet-W48 sensitivity
# sweep showed one model can move 16 px across checkpoints -- more than the gap
# between architectures. A fixed-epoch comparison removes that dependency
# entirely: every model is read at the same epoch, chosen without consulting the
# validation labels.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=/Users/lizhiheng/miniforge3/envs/DEEPLABCUT/bin/python
LOG=local_data/experiments/03_architecture_sweep_batch2/checkpoint_matrix.log
export MPLCONFIGDIR="$PWD/local_data/matplotlib" NUMBA_CACHE_DIR="$PWD/local_data/numba"

# index 0..4 maps to snapshot-100/125/150/175/200 for every shuffle here,
# because all five models kept exactly those numbered snapshots and only some
# also kept a best snapshot (which sorts last and is not indexed below).
declare -a MODELS=("hrnet_w32:6" "hrnet_w18:10" "resnet_50:11" "hrnet_w48:13" "cspnext_s:14")
declare -a EPOCHS=("0:100" "1:125" "2:150" "3:175" "4:200")

echo "[$(date -Iseconds)] START checkpoint matrix" >> "$LOG"
for entry in "${MODELS[@]}"; do
  model="${entry%%:*}"; shuffle="${entry#*:}"
  for spec in "${EPOCHS[@]}"; do
    idx="${spec%%:*}"; epoch="${spec#*:}"
    label="matrix_${model}_ep${epoch}"
    dir="local_data/test_sets/eye_last_minute_100/predictions_100train_${label}"
    if [[ -f "$dir/eye_test_summary.csv" ]]; then
      echo "[$(date -Iseconds)] SKIP $label (already evaluated)" >> "$LOG"
      continue
    fi
    echo "[$(date -Iseconds)] eval $label (shuffle $shuffle, index $idx)" >> "$LOG"
    $PY scripts/evaluate_eye_test_set.py --train-frames 100 --shuffle "$shuffle" \
        --model-label "$label" --snapshot-index "$idx" >> "$LOG" 2>&1
    echo "[$(date -Iseconds)] EXIT: $? ($label)" >> "$LOG"
  done
done
echo "[$(date -Iseconds)] DONE checkpoint matrix" >> "$LOG"
