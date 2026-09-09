#!/bin/bash
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/local_data/experiments/02_batch32_extension"
LOG="$OUT/batch32_shuffle9.log"

echo "experiment: $OUT"
if [[ -f "$OUT/background.pid" ]]; then
  PID="$(cat "$OUT/background.pid")"
  echo "pid: $PID"
  if kill -0 "$PID" 2>/dev/null; then
    echo "process: running"
  else
    echo "process: stopped"
  fi
else
  echo "pid: not started"
fi

if [[ -f "$LOG" ]]; then
  echo
  echo "latest training output:"
  tr '\r' '\n' < "$LOG" |
    grep -E 'Epoch [0-9]+/200|Traceback|Error|failed|EXIT:' |
    tail -n 8
fi

if [[ -f "$OUT/run_metadata.csv" ]]; then
  echo
  cat "$OUT/run_metadata.csv"
fi
