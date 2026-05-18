# MLIR / NPU Compiler Course - Lecture 05 Handout

## Topic
Dialect System - frontend / graph / kernel / runtime dialect separation.

## Core idea
MLIR Dialect는 단순 namespace가 아니라 compiler pass, verifier, lowering, runtime ABI가 공유하는 semantic contract입니다.

```text
Frontend dialects  : StableHLO / TOSA / ONNX import semantics
Graph dialect      : fusion, layout propagation, quant metadata, partitioning
Kernel dialect     : tiled compute, DMA, barrier, scratchpad, accumulator
Runtime dialect    : command buffer, tensor descriptor, launch/sync/profiling
```

## Dialect anatomy

```text
Dialect
  - namespace
  - operations
  - types
  - attributes
  - traits
  - interfaces
  - verifiers
  - parser / printer
  - canonicalization / conversion patterns
```

## Design rules for NPU compiler

1. Keep frontend semantics as long as useful.
2. Reuse common dialects such as tensor/linalg/memref/scf/vector before inventing custom target ops.
3. Introduce `npu_graph` when fusion/layout/quant decisions become target-specific.
4. Introduce `npu_kernel` when tile, SRAM, DMA, barrier, accumulator details become explicit.
5. Introduce `npu_rt` when command buffer ABI, tensor descriptors, launch and sync appear.
6. Put runtime values in operands, static metadata in attributes, ABI/shape/layout contracts in types.
7. Every target-specific op should have verifier rules and FileCheck tests.

## Example boundary

```text
stablehlo.dot_general + add + relu
  -> linalg.matmul + linalg.generic
  -> npu_graph.fusion_group @matmul_bias_relu
  -> npu_kernel.region { dma_load; barrier; matmul; requant_relu; dma_store }
  -> npu_rt.launch command_buffer
```

## Checklist for custom NPU dialect

- What is the dialect namespace?
- Which abstraction level does it represent?
- What operations are legal in this dialect?
- Which types and attributes are first-class?
- Which traits and interfaces should passes depend on?
- What should the verifier reject?
- What is the next lowering target?
- What are the regression tests?

## References
- MLIR Language Reference: https://mlir.llvm.org/docs/LangRef/
- Defining Dialects: https://mlir.llvm.org/docs/DefiningDialects/
- Operation Definition Specification: https://mlir.llvm.org/docs/DefiningDialects/Operations/
- Dialect Conversion: https://mlir.llvm.org/docs/DialectConversion/
