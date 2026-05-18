#!/usr/bin/env bash
set -euo pipefail

# Example commands. Adjust pass availability to your LLVM/MLIR checkout.
mlir-opt 02_linalg_generic_matmul_tensor.mlir --canonicalize --cse
mlir-opt 06_filecheck_template.mlir --canonicalize --cse | FileCheck 06_filecheck_template.mlir
