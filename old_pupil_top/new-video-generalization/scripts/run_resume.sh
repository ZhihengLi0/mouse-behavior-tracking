#!/bin/bash
# Resume of the 2026-09-19 night plan after a user stop: step 4 is already trained and scored,
# so only analyze the video with its FINAL snapshot, select batch 5, then run the seed-43 replicates.
UNIT=/Users/lizhiheng/Desktop/生物/new-video-generalization
cd "$UNIT/scripts"
PY=/Users/lizhiheng/miniforge3/envs/DEEPLABCUT/bin/python; N=/tmp/scale_night.log; LOG=/tmp/scale_step04.log
export PYTORCH_ENABLE_MPS_FALLBACK=1
SHUF=84; TSI=10; FIN=8
say(){ echo "[$(date '+%m-%d %H:%M')] $*" >> $N; }
say "=== RESUME: step-4 video analysis (shuffle $SHUF, tsi $TSI, final index $FIN) ==="
DEST="$UNIT/training-data/predictions_step04"; mkdir -p "$DEST"
$PY - >> "$LOG" 2>&1 <<PYEOF
import deeplabcut
deeplabcut.analyze_videos(r"/Users/lizhiheng/Desktop/生物/dlc_projects/EyePupilBlink-Zhiheng-2026-08-17/config.yaml",
    [r"$UNIT/training-data/20251031/20251031_Pluto_spont_1.mp4"], shuffle=$SHUF, trainingsetindex=$TSI, snapshot_index=$FIN,
    device="mps", batch_size=32, destfolder=r"$DEST")
PYEOF
H5=$(ls "$DEST"/*shuffle${SHUF}*snapshot_100*.h5 2>/dev/null | tail -1)
if [ -n "$H5" ]; then
  $PY select_scale_frames.py --stage batch --batch-no 5 --pred-h5 "$H5" >> "$LOG" 2>&1 && say "BATCH05 READY" || say "BATCH05 SELECTION FAILED"
else say "no predictions h5 for step 4"; fi
for S in 1 2 3; do
  say "=== seed-43 replicate of step $S (shuffle 8$S) ==="
  $PY scale_step.py --step $S --shuffle 8$S --seed 43 --tag seed43 > /tmp/scale_rep0$S.log 2>&1 && say "replicate $S done" || say "REPLICATE $S FAILED"
done
say "=== NIGHT DONE ==="
