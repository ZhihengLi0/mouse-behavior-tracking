#!/bin/bash
# One scale step for one video: train -> evaluate (if test50 is labeled) -> predict the whole video with the FINAL
# snapshot -> select the next batch (jump rule + k-means) -> draw its cluster + time-series sheet.
#   ./run_step.sh <unit> <video path> <step> <shuffle> [prior ...]
set -u
UNIT=$1; VIDEO=$2; STEP=$3; SHUF=$4; shift 4; PRIOR="$*"; NEXT=$((STEP+1))
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/Users/lizhiheng/miniforge3/envs/DEEPLABCUT/bin/python
LOG=/tmp/newstd_${UNIT}_step$(printf %02d $STEP).log
CONFIG="$HERE/../dlc_projects/EyePupilEllipse-Zhiheng-2026-09-20/config.yaml"
export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "$HERE/scripts"
say(){ echo "[$(date '+%m-%d %H:%M')] $*" >> "$LOG"; }
say "=== $UNIT step $STEP (shuffle $SHUF) ==="
if $PY scale_step.py --unit "$UNIT" --step $STEP --shuffle $SHUF ${PRIOR:+--prior $PRIOR} >> "$LOG" 2>&1; then
  TSI=$(grep -a -o 'TSI=[0-9]*' "$LOG" | tail -1 | cut -d= -f2); FIN=$(grep -a -o 'FINAL_INDEX=[0-9]*' "$LOG" | tail -1 | cut -d= -f2)
  say "step done (tsi $TSI); predicting the whole video with the FINAL snapshot (index $FIN)"
  DEST="$HERE/$UNIT/training-data/predictions_step$(printf %02d $STEP)"; mkdir -p "$DEST"
  $PY - >> "$LOG" 2>&1 <<PYEOF
import deeplabcut
deeplabcut.analyze_videos(r"$CONFIG", [r"$VIDEO"], shuffle=$SHUF, trainingsetindex=$TSI, snapshot_index=$FIN,
    device="mps", batch_size=32, destfolder=r"$DEST")
PYEOF
  H5=$(ls "$DEST"/*shuffle${SHUF}*snapshot_120*.h5 2>/dev/null | tail -1)
  if [ -n "$H5" ]; then
    if $PY select_frames.py --unit "$UNIT" --video "$VIDEO" --stage batch --batch-no $NEXT --pred-h5 "$H5" >> "$LOG" 2>&1; then
      say "BATCH$(printf %02d $NEXT) READY"
      $PY viz_selection_timeline.py --unit "$UNIT" --batch-no $NEXT >> "$LOG" 2>&1 && say "selection sheet drawn"
    else say "NEXT BATCH SELECTION FAILED"; fi
  else say "no predictions h5"; fi
else say "STEP FAILED"; fi
say "=== ALL DONE ==="
