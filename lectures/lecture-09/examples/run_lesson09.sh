#!/usr/bin/env bash
set -euo pipefail
MLIR_OPT=${MLIR_OPT:-mlir-opt}

$MLIR_OPT 01_pass_pipeline_input.mlir \
  --pass-pipeline='builtin.module(func.func(cse,canonicalize))'

$MLIR_OPT 01_pass_pipeline_input.mlir \
  --pass-pipeline='builtin.module(func.func(cse,canonicalize))' \
  --mlir-print-ir-after-change --mlir-timing --mlir-pass-statistics

# Once custom NPU passes exist, run:
# $MLIR_OPT 01_pass_pipeline_input.mlir --pass-pipeline='builtin.module(npu-lowering-pipeline{sram-kb=512 tile-m=64 tile-n=128 tile-k=32})'
