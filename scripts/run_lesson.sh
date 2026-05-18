#!/usr/bin/env bash
set -u

usage() {
  cat >&2 <<'USAGE'
usage: scripts/run_lesson.sh <lecture-number> [--try-mlir] [--strict]

Runs the stock-tool portion of a lecture lab.

Notes:
  - This course intentionally contains pseudo NPU dialects and custom pass
    examples. Those files are reported as SKIP unless a local NPU backend is
    installed.
  - Stock MLIR files are parse/verify checked with mlir-opt.
  - Python helper scripts in examples/ are executed.
USAGE
}

if [ $# -lt 1 ]; then
  usage
  exit 2
fi

RAW_LECTURE="$1"
shift || true
case "$RAW_LECTURE" in
  ''|*[!0-9]*) echo "lecture number must be numeric: $RAW_LECTURE" >&2; exit 2 ;;
esac
LECTURE=$(printf "%02d" "$((10#$RAW_LECTURE))")

TRY_MLIR=0
STRICT=0
for arg in "$@"; do
  case "$arg" in
    --try-mlir) TRY_MLIR=1 ;;
    --strict) STRICT=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $arg" >&2; usage; exit 2 ;;
  esac
done

ROOT=$(cd "$(dirname "$0")/.." && pwd)
DIR="$ROOT/lectures/lecture-$LECTURE"
if [ ! -d "$DIR" ]; then
  echo "lecture not found: $DIR" >&2
  exit 1
fi

MLIR_OPT=${MLIR_OPT:-mlir-opt}
PYTHON=${PYTHON:-python3}
STATUS=0
CHECKED=0
SKIPPED=0
RAN_PY=0

has_custom_or_pseudo_content() {
  local file="$1"
  grep -Eq '(^|[^[:alnum:]_])(stablehlo\.|npu(_[a-zA-Z0-9]+)?\.|npu-opt|--npu-|#npu\.|!npu\.|transform\.npu\.|!npu_|#npu_|npu_graph|npu_kernel|npu_rt|quant\.qcast|quant\.scast|tosa\.rounding_mode|tile_using_for|pdl\.operation|pdl\.rewrite|custom NPU backend|pseudo-MLIR|pseudo MLIR|Pseudo NPU|Pseudo custom)' "$file"
}

uses_stock_incompatible_run_line() {
  local file="$1"
  grep -Eq '^// RUN:.*(npu-opt|--npu-|convert-npu|npu-[a-zA-Z0-9_-]+)' "$file"
}

run_mlir_file() {
  local file="$1"
  local rel="${file#$DIR/}"

  if has_custom_or_pseudo_content "$file" || uses_stock_incompatible_run_line "$file"; then
    echo "-- SKIP $rel (pseudo/custom dialect or custom NPU pass example)"
    SKIPPED=$((SKIPPED + 1))
    return 0
  fi

  local cmd=("$MLIR_OPT" --allow-unregistered-dialect --verify-diagnostics)
  if grep -q '^// -----$' "$file"; then
    cmd+=(--split-input-file)
  fi
  cmd+=("$file")

  echo "-- CHECK $rel"
  if "${cmd[@]}" >/dev/null; then
    echo "   ok"
    CHECKED=$((CHECKED + 1))
  else
    echo "   failed: $rel"
    "${cmd[@]}" >/dev/null
    STATUS=1
  fi
}

run_python_file() {
  local file="$1"
  local rel="${file#$DIR/}"
  local base
  base=$(basename "$file")
  if [[ "$base" == lit*.cfg.py || "$base" == *local.cfg.py ]]; then
    echo "-- SKIP $rel (lit configuration, not a standalone lab script)"
    SKIPPED=$((SKIPPED + 1))
    return 0
  fi
  echo "-- PYTHON $rel"
  if (cd "$(dirname "$file")" && "$PYTHON" "$(basename "$file")") >/dev/null; then
    echo "   ok"
    RAN_PY=$((RAN_PY + 1))
  else
    echo "   failed: $rel"
    (cd "$(dirname "$file")" && "$PYTHON" "$(basename "$file")")
    STATUS=1
  fi
}

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
  echo "Local runner: ${RUNNER#$DIR/} (manual/reference; root runner performs stock-safe checks below)"
fi

if [ "$TRY_MLIR" -eq 0 ]; then
  echo "Tip: add --try-mlir to run stock-safe MLIR checks and Python helpers."
  exit 0
fi

if ! command -v "$MLIR_OPT" >/dev/null 2>&1; then
  echo "mlir-opt not found: $MLIR_OPT" >&2
  echo "Set PATH or MLIR_OPT. See docs/lab-environment.md" >&2
  if [ "$STRICT" -eq 1 ]; then exit 1; fi
  exit 0
fi

while IFS= read -r file; do
  run_mlir_file "$file"
done < <(find "$DIR/examples" -maxdepth 1 -type f -name '*.mlir' 2>/dev/null | sort)

while IFS= read -r file; do
  run_python_file "$file"
done < <(find "$DIR/examples" -maxdepth 1 -type f -name '*.py' 2>/dev/null | sort)

echo "Summary: checked=$CHECKED skipped=$SKIPPED python=$RAN_PY status=$STATUS"

if [ "$STRICT" -eq 1 ]; then
  exit "$STATUS"
fi
exit 0
