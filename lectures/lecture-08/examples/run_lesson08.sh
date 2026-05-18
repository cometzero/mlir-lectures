#!/usr/bin/env bash
set -euo pipefail

mlir-opt 01_scf_for_basic.mlir --canonicalize --cse
mlir-opt 03_affine_tiled_matmul.mlir --canonicalize --affine-loop-normalize
mlir-opt 06_filecheck_template.mlir --split-input-file --canonicalize | FileCheck 06_filecheck_template.mlir
python3 07_tile_size_sram_calculator.py
