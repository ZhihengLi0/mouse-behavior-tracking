#!/bin/bash
# Night of 2026-09-19: step 4 (then batch 5 selection), then second-seed replicates of steps 1-3.
cd /Users/lizhiheng/Desktop/生物/new-video-generalization/scripts
PY=/Users/lizhiheng/miniforge3/envs/DEEPLABCUT/bin/python; N=/tmp/scale_night.log
export PYTORCH_ENABLE_MPS_FALLBACK=1
say(){ echo "[$(date '+%m-%d %H:%M')] $*" >> $N; }
say "=== NIGHT START ==="
rm -f /tmp/scale_step04.log; ./run_step.sh 4 84; say "step 4 driver finished: $(LC_ALL=C grep -a 'READY\|FAILED' /tmp/scale_step04.log | tail -1)"
for S in 1 2 3; do
  say "=== seed-43 replicate of step $S (shuffle 8$S) ==="
  $PY scale_step.py --step $S --shuffle 8$S --seed 43 --tag seed43 > /tmp/scale_rep0$S.log 2>&1 && say "replicate $S done" || say "REPLICATE $S FAILED"
done
say "=== NIGHT DONE ==="
