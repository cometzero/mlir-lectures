# MLIR NPU Lecture 20 Handout - Capstone

## Goal
Design a mini MLIR-based NPU compiler and submit a design spec that connects input IR, transformation pipeline, custom NPU dialect, memory planning, runtime ABI, and verification.

## Required Pipeline
```text
StableHLO/TOSA -> npu_graph -> Linalg/Tensor -> Transform Schedule -> Bufferization/MemRef -> npu_kernel -> npu_rt -> Command Buffer
```

## Minimum Deliverables
1. Mini NPU Compiler Design Spec
2. Pass pipeline and legal op set
3. Pseudo MLIR before/after examples
4. NPU dialect contract for matmul/dma/barrier
5. Runtime descriptor and command buffer ABI
6. Verification plan with positive/negative tests
7. Simple cost model and presentation

## Design Rule
Every compiler decision must be tied to one of: correctness, target legality, memory capacity, numerical behavior, performance, or runtime ABI compatibility.
