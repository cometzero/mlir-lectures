#!/usr/bin/env bash
set -euo pipefail

# These commands assume a custom NPU backend has registered the pseudo passes.
mlir-opt 04_npu_rt_dialect_pseudo.mlir   --npu-rt-verify-abi   --npu-rt-pack-command-buffer   --npu-rt-emit-artifact-manifest

mlir-opt 08_filecheck_runtime_abi_test.mlir   --npu-rt-verify-abi --npu-rt-pack-command-buffer | FileCheck 08_filecheck_runtime_abi_test.mlir
