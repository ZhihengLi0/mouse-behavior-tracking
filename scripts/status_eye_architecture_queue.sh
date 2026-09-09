#!/bin/bash
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/local_data/experiments/03_architecture_sweep_batch2"

echo "experiment: $OUT"
if [[ -f "$OUT/background.pid" ]]; then
  PID="$(cat "$OUT/background.pid")"
  echo "pid: $PID"
  kill -0 "$PID" 2>/dev/null && echo "process: running" || echo "process: stopped"
else
  echo "process: waiting for initial HRNet-W18 run"
fi
[[ -f "$OUT/run_metadata.csv" ]] && cat "$OUT/run_metadata.csv"
for log in "$OUT"/*_shuffle*.log; do
  [[ -e "$log" ]] || continue
  echo
  echo "--- $(basename "$log") ---"
  tr '\r' '\n' < "$log" |
    grep -E 'Epoch [0-9]+/200|Traceback|Error|failed|EXIT:' |
    tail -n 5
done
