#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "最原始的视频body1h，eye5min/face.mp4"
ffprobe -v error \
  -show_entries format=duration,size:stream=index,codec_type,codec_name,width,height,r_frame_rate \
  -of default=noprint_wrappers=1 \
  face.mp4

echo
echo "最原始的视频body1h，eye5min/body.mp4"
ffprobe -v error \
  -show_entries format=duration,size:stream=index,codec_type,codec_name,width,height,r_frame_rate \
  -of default=noprint_wrappers=1 \
  body.mp4
