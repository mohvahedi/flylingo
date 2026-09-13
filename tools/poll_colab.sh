#!/usr/bin/env bash
# Poll the Colab run without blocking, and keep the session warm.
#
# Two jobs. It appends progress to a local log so the run is inspectable while it happens, and it
# makes a short CLI call every cycle, which keeps the session active from the client side (a Colab
# VM is reclaimed when idle, and this run must not be).
#
# Exits when the run reports ALL DONE or raises, so the caller gets one notification.
set -u
PY="C:/c/Users/Lion/.local/colab-cli-venv/Scripts/python.exe"
LOG="D:/Projects/flylingo/artifacts/colab_run.log"
SESSION="${1:-gpu}"
INTERVAL="${2:-90}"
MAX_CYCLES="${3:-60}"

mkdir -p "$(dirname "$LOG")"
: > "$LOG"

for i in $(seq 1 "$MAX_CYCLES"); do
  printf 'import os\np="/content/run_all.log"\nprint(open(p).read()[-1500:] if os.path.exists(p) else "NO LOG YET")\n' \
    | "$PY" -m colab_cli.cli exec -s "$SESSION" > "$LOG.tmp" 2>&1
  {
    echo "===== cycle $i at $(date +%H:%M:%S) ====="
    tail -c 1800 "$LOG.tmp"
  } >> "$LOG"

  if grep -qE "ALL DONE" "$LOG.tmp"; then
    echo "RUN COMPLETE" >> "$LOG"; break
  fi
  if grep -qE "Traceback|SystemExit|EQUIVALENCE GATE FAILED" "$LOG.tmp"; then
    echo "RUN FAILED" >> "$LOG"; break
  fi
  sleep "$INTERVAL"
done
tail -c 3000 "$LOG"
