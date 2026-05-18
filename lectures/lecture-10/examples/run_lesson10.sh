#!/usr/bin/env bash
set -euo pipefail

# Examples assume an MLIR build with the teaching pass registered.
# Use these commands as templates for your own NPU compiler tree.

mlir-opt 01_linalg_matmul_add_relu_before.mlir \
  --canonicalize --cse \
  --npu-fuse-epilogue \
  --mlir-print-ir-after-change

mlir-opt 06_filecheck_fusion_test.mlir --npu-fuse-epilogue | FileCheck 06_filecheck_fusion_test.mlir
mlir-opt 07_negative_no_fusion_test.mlir --npu-fuse-epilogue | FileCheck 07_negative_no_fusion_test.mlir
