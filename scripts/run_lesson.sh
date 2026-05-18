#!/usr/bin/env bash
set -u
if [ $# -lt 1 ]; then
  echo "usage: scripts/run_lesson.sh <lecture-number> [--try-mlir] [--strict]" >&2
  exit 2
fi
LECTURE=$(printf "%02d" "$1")
shift || true
TRY_MLIR=0
STRICT=0
for arg in "$@"; do
  case "$arg" in
    --try-mlir) TRY_MLIR=1 ;;
    --strict) STRICT=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DIR="$ROOT/lectures/lecture-$LECTURE"
if [ ! -d "$DIR" ]; then
  echo "lecture not found: $DIR" >&2
  exit 1
fi
echo "== Lecture $LECTURE =="
echo "Directory: $DIR"
if [ -f "$DIR/LESSON.md" ]; then
  echo "Guide: $DIR/LESSON.md"
fi
if [ -d "$DIR/examples" ]; then
  echo "Examples:"
  find "$DIR/examples" -maxdepth 1 -type f | sort | sed "s#^$DIR/#  #"
else
  echo "No examples directory."
fi
RUNNER=$(find "$DIR/examples" -maxdepth 1 -type f -name 'run_lesson*.sh' 2>/dev/null | sort | head -n 1 || true)
if [ -n "$RUNNER" ]; then
  echo "Found lesson runner: ${RUNNER#$DIR/}"
fi
if [ "$TRY_MLIR" -eq 0 ]; then
  echo "Tip: add --try-mlir to attempt mlir-opt on .mlir examples."
  exit 0
fi
if ! command -v mlir-opt >/dev/null 2>&1; then
  echo "mlir-opt not found. See docs/lab-environment.md" >&2
  exit 0
fi
STATUS=0
while IFS= read -r file; do
  echo "-- mlir-opt --allow-unregistered-dialect --verify-diagnostics ${file#$DIR/}"
  if ! mlir-opt --allow-unregistered-dialect --verify-diagnostics "$file" >/dev/null; then
    echo "   failed: ${file#$DIR/}"
    STATUS=1
  else
    echo "   ok"
  fi
done < <(find "$DIR/examples" -maxdepth 1 -type f -name '*.mlir' 2>/dev/null | sort)
if [ "$STRICT" -eq 1 ]; then
  exit "$STATUS"
fi
exit 0
