// Lecture 14 - conceptual pipeline snippets

// RUN: mlir-opt %s \
// RUN:   --pass-pipeline='builtin.module(func.func(canonicalize,cse,linalg-vectorize,npu-vector-contract-tensorize,npu-vector-transfer-legalize))' \
// RUN:   | FileCheck %s

// Pipeline intent:
// 1. Keep linalg structured semantics long enough for fusion/tiling.
// 2. Vectorize into vector.transfer_read/write and vector.contract.
// 3. Tensorize vector.contract if it matches native NPU tile ISA.
// 4. Legalize transfer ops to DMA, SRAM view, or fallback path.
