#!/usr/bin/env bash
set -euo pipefail

# Example commands. Requires a local MLIR build with the relevant passes.
MLIR_OPT=${MLIR_OPT:-mlir-opt}
FILECHECK=${FILECHECK:-FileCheck}

$MLIR_OPT 02_vector_contract_matmul.mlir \
  --canonicalize --cse \
  --mlir-print-ir-after-change

# Pseudo custom pass test:
# $MLIR_OPT 07_filecheck_vector_test.mlir --npu-vector-contract-tensorize | $FILECHECK 07_filecheck_vector_test.mlir
