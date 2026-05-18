#!/usr/bin/env bash
# Lecture 04 command recipes. Assumes MLIR tools are in PATH.
set -euo pipefail

mlir-opt 01_parse_verify.mlir | FileCheck 01_parse_verify.mlir
mlir-opt 02_cse_regression_test.mlir --pass-pipeline='builtin.module(func.func(cse))' | FileCheck 02_cse_regression_test.mlir
mlir-opt 03_pass_pipeline_debug.mlir --pass-pipeline='builtin.module(func.func(cse,canonicalize))' --mlir-print-ir-after-all
mlir-opt 04_filecheck_directives.mlir --pass-pipeline='builtin.module(func.func(cse))' | FileCheck 04_filecheck_directives.mlir
mlir-opt 05_verify_diagnostics_negative.mlir -split-input-file -verify-diagnostics
mlir-translate --mlir-to-llvmir 06_mlir_translate_to_llvm.mlir | FileCheck 06_mlir_translate_to_llvm.mlir
