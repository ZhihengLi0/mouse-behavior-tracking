#!/bin/bash
# Driver for one scale step: train from scratch, evaluate, analyze the video, select the next batch.
#   ./run_step.sh <step> <shuffle>
set -u
STEP=$1; SHUF=$2; NEXT=$((STEP+1))
UNIT=/Users/lizhiheng/Desktop/生物/new-video-generalization
PY=/Users/lizhiheng/miniforge3/envs/DEEPLABCUT/bin/python
LOG=/tmp/scale_step$(printf %02d $STEP).log
export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "$UNIT/scripts"
say(){ echo "[$(date '+%m-%d %H:%M')] $*" >> "$LOG"; }
say "=== step $STEP: from scratch, shuffle $SHUF ==="
if $PY scale_step.py --step $STEP --shuffle $SHUF >> "$LOG" 2>&1; then
  TSI=$(grep -o 'TSI=[0-9]*' "$LOG" | tail -1 | cut -d= -f2); FIN=$(grep -o 'FINAL_INDEX=[0-9]*' "$LOG" | tail -1 | cut -d= -f2)
  say "step done (tsi $TSI); analyzing the video with the FINAL snapshot (index $FIN)"
  DEST="$UNIT/training-data/predictions_step$(printf %02d $STEP)"; mkdir -p "$DEST"
  $PY - >> "$LOG" 2>&1 <<PYEOF
import deeplabcut
deeplabcut.analyze_videos(r"/Users/lizhiheng/Desktop/生物/dlc_projects/EyePupilBlink-Zhiheng-2026-08-17/config.yaml",
    [r"$UNIT/training-data/20251031/20251031_Pluto_spont_1.mp4"], shuffle=$SHUF, trainingsetindex=$TSI, snapshot_index=$FIN,
    device="mps", batch_size=32, destfolder=r"$DEST")
PYEOF
  H5=$(ls "$DEST"/*shuffle${SHUF}*snapshot_100*.h5 2>/dev/null | tail -1)
  if [ -n "$H5" ]; then
    $PY select_scale_frames.py --stage batch --batch-no $NEXT --pred-h5 "$H5" >> "$LOG" 2>&1 && say "BATCH$(printf %02d $NEXT) READY" || say "NEXT BATCH SELECTION FAILED"
  else say "no predictions h5"; fi
else say "STEP FAILED"; fi
say "=== ALL DONE ==="
