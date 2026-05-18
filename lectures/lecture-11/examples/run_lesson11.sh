#!/usr/bin/env bash
set -euo pipefail

# Example commands. Replace `--npu-legalize-graph` with your actual pass name.
mlir-opt 02_before_unsupported_ops.mlir --canonicalize --npu-legalize-graph
mlir-opt 07_filecheck_conversion_test.mlir --npu-legalize-graph | FileCheck 07_filecheck_conversion_test.mlir
mlir-opt 08_negative_unsupported_shape_test.mlir --npu-legalize-graph -verify-diagnostics
