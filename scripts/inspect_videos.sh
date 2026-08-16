#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "face.mp4"
ffprobe -v error \
  -show_entries format=duration,size:stream=index,codec_type,codec_name,width,height,r_frame_rate \
  -of default=noprint_wrappers=1 \
  face.mp4

echo
echo "body.mp4"
ffprobe -v error \
  -show_entries format=duration,size:stream=index,codec_type,codec_name,width,height,r_frame_rate \
  -of default=noprint_wrappers=1 \
  body.mp4
