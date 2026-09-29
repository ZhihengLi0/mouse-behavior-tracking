#!/bin/bash
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/batch-size-selection/logs"
LOG="$OUT/resource_watchdog.log"
LOW_COUNT=0

while true; do
  [[ -f "$OUT/background.pid" ]] || { sleep 60; continue; }
  PID="$(cat "$OUT/background.pid")"
  kill -0 "$PID" 2>/dev/null || break

  FREE_PERCENT="$(
    memory_pressure |
      awk '/System-wide memory free percentage/ {gsub(/%/, "", $5); print $5}'
  )"
  DISK_FREE_KB="$(df -k "$ROOT" | awk 'NR == 2 {print $4}')"
  CHILD_RSS_KB="$(
    ps -axo ppid=,rss= |
      awk -v parent="$PID" '$1 == parent {sum += $2} END {print sum + 0}'
  )"
  printf '%s free_memory=%s%% child_rss_kb=%s disk_free_kb=%s\n' "$(date -Iseconds)" "$FREE_PERCENT" "$CHILD_RSS_KB" "$DISK_FREE_KB" >> "$LOG"

  if [[ "$FREE_PERCENT" -lt 10 || "$CHILD_RSS_KB" -gt 10485760 ]]; then
    LOW_COUNT=$((LOW_COUNT + 1))
  else
    LOW_COUNT=0
  fi

  if [[ "$LOW_COUNT" -ge 3 || "$DISK_FREE_KB" -lt 20971520 ]]; then
    printf '%s STOPPING queue: resource safety threshold reached\n' "$(date -Iseconds)" >> "$LOG"
    pkill -TERM -P "$PID" 2>/dev/null || true
    kill -TERM "$PID" 2>/dev/null || true
    exit 2
  fi
  sleep 60
done

printf '%s queue exited; watchdog complete\n' "$(date -Iseconds)" >> "$LOG"
