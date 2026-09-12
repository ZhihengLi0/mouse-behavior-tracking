#!/usr/bin/env bash
# Open one active-learning branch's 20 frames for human review.
# Usage: bash label_branch.sh {uncertain|jump|fitting}
set -euo pipefail
BRANCH="${1:?usage: label_branch.sh {uncertain|jump|fitting}}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="/Users/lizhiheng/miniforge3/envs/DEEPLABCUT/bin/python"
CONFIG="$PROJECT_DIR/dlc_projects/EyePupilBlink-Zhiheng-2026-08-17/config.yaml"
FOLDER="$PROJECT_DIR/active-learning/frames/$BRANCH"
[[ -d "$FOLDER" ]] || { echo "no such branch folder: $FOLDER"; exit 1; }
mkdir -p "$PROJECT_DIR/local_data/matplotlib" "$PROJECT_DIR/local_data/numba" \
         "$PROJECT_DIR/local_data/xdg_cache" "$PROJECT_DIR/local_data/napari_config"
export MPLCONFIGDIR="$PROJECT_DIR/local_data/matplotlib"
export NUMBA_CACHE_DIR="$PROJECT_DIR/local_data/numba"
export XDG_CACHE_HOME="$PROJECT_DIR/local_data/xdg_cache"
export NAPARI_CONFIG="$PROJECT_DIR/local_data/napari_config/settings.yaml"
"$PYTHON" - <<PY
from deeplabcut.gui.widgets import launch_napari
import napari
viewer = launch_napari(files=["$FOLDER", "$CONFIG"], plugin="napari-deeplabcut", stack=False)
napari.run()
PY
