#!/usr/bin/env bash
set -euo pipefail

MLIR_OPT=${MLIR_OPT:-mlir-opt}

${MLIR_OPT} 02_linalg_on_tensors_before_bufferization.mlir \
  -canonicalize -cse \
  -one-shot-bufferize="allow-unknown-ops" \
  -canonicalize

${MLIR_OPT} 08_filecheck_bufferization_test.mlir \
  -one-shot-bufferize -canonicalize | FileCheck 08_filecheck_bufferization_test.mlir
