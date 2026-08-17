#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/Users/lizhiheng/miniforge3/envs/DEEPLABCUT/bin/python"

mkdir -p "$PROJECT_DIR/local_data/matplotlib"
export MPLCONFIGDIR="$PROJECT_DIR/local_data/matplotlib"

echo "Project: $PROJECT_DIR"
echo

echo "Conda environments:"
/Users/lizhiheng/miniforge3/bin/conda info --envs
echo

echo "DeepLabCut/PyTorch:"
"$PYTHON" - <<'PY'
import deeplabcut
import torch
import torchvision

print("deeplabcut", deeplabcut.__version__)
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("mps", torch.backends.mps.is_available())
PY
echo

echo "Git status:"
git -C "$PROJECT_DIR" status --short --branch
echo

echo "Ignored raw files:"
git -C "$PROJECT_DIR" check-ignore -v \
  face.mp4 \
  body.mp4 \
  13336_Inferring_how_internal_b.pdf \
  s41593-023-01490-6.pdf
