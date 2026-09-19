#!/bin/bash
# Overnight driver for scale step 1 (2026-09-19). Sequential, logged, sleep-proof.
set -u
UNIT=/Users/lizhiheng/Desktop/生物/new-video-generalization
PY=/Users/lizhiheng/miniforge3/envs/DEEPLABCUT/bin/python
LOG=/tmp/scale_step01.log
export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "$UNIT/scripts"
say(){ echo "[$(date '+%m-%d %H:%M')] $*" >> "$LOG"; }

say "=== baseline (production_v1 unadapted) ==="
$PY scale_step.py --baseline >> "$LOG" 2>&1 || say "BASELINE FAILED"

say "=== arm A: scratch, shuffle 61 ==="
if $PY scale_step.py --step 1 --arm scratch --shuffle 61 >> "$LOG" 2>&1; then
  TSI=$(grep -o 'TSI=[0-9]*' "$LOG" | tail -1 | cut -d= -f2)
  say "arm A done (tsi $TSI); analyzing the Pluto video with the new model"
  mkdir -p "$UNIT/training-data/predictions_step01"
  $PY - >> "$LOG" 2>&1 <<PYEOF
import deeplabcut
deeplabcut.analyze_videos(r"/Users/lizhiheng/Desktop/生物/dlc_projects/EyePupilBlink-Zhiheng-2026-08-17/config.yaml",
    [r"$UNIT/training-data/20251031/20251031_Pluto_spont_1.mp4"], shuffle=61, trainingsetindex=$TSI,
    device="mps", batch_size=32, destfolder=r"$UNIT/training-data/predictions_step01")
PYEOF
  H5=$(ls "$UNIT"/training-data/predictions_step01/*shuffle61*.h5 2>/dev/null | tail -1)
  if [ -n "$H5" ]; then
    say "selecting batch02 with the jump detector"
    $PY select_scale_frames.py --stage batch --batch-no 2 --pred-h5 "$H5" >> "$LOG" 2>&1 && say "BATCH02 READY" || say "BATCH02 SELECTION FAILED"
  else
    say "no predictions h5 - batch02 not selected"
  fi
else
  say "ARM A FAILED"
fi

say "=== arm B: warm start from production_v1, shuffle 62 ==="
$PY scale_step.py --step 1 --arm warm --shuffle 62 >> "$LOG" 2>&1 && say "arm B done" || say "ARM B FAILED (exploratory arm; main line unaffected)"
say "=== ALL DONE ==="
