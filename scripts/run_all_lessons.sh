#!/usr/bin/env bash
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
STATUS=0
for n in $(seq 1 20); do
  printf '\n===== Lecture %02d =====\n' "$n"
  if ! "$ROOT/scripts/run_lesson.sh" "$n" --try-mlir --strict; then
    STATUS=1
  fi
done
exit "$STATUS"
