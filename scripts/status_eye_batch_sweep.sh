#!/bin/bash
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/local_data/experiments/01_batch_size_sweep"

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

if [[ -f "$OUT/run_metadata.csv" ]]; then
  echo
  echo "completed run metadata:"
  cat "$OUT/run_metadata.csv"
fi

echo
echo "latest logs:"
for log in "$OUT"/*.log; do
  [[ -e "$log" ]] || continue
  echo "--- $(basename "$log") ---"
  tail -n 8 "$log"
done
