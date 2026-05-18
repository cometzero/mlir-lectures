#!/usr/bin/env bash
set -euo pipefail
# Example usage. Paths are placeholders for your local MLIR build.
MLIR_OPT=${MLIR_OPT:-mlir-opt}

${MLIR_OPT} 01_payload_linalg_matmul_add_relu.mlir \
  --transform-interpreter \
  --canonicalize \
  --mlir-print-ir-after-change \
  -o /tmp/l15_after.mlir

FileCheck 05_filecheck_transform_test.mlir < /tmp/l15_after.mlir
