# MLIR NPU Lecture 12 Handout

## Topic
Bufferization: tensor -> memref, alias/lifetime analysis for NPU compilers.

## Core mental model

```text
Tensor/Linalg world
  - immutable SSA values
  - fusion/tiling/canonicalization friendly
  - no physical storage commitment yet

Bufferization boundary
  - decide in-place vs copy
  - create memref/storage semantics
  - expose alloc/copy/dealloc and alias/lifetime issues

MemRef/NPU world
  - DRAM/SRAM/ACCUM memory spaces
  - DMA scheduling and command buffer ABI
  - explicit descriptors and lifetimes
```

## Key terms

| Term | Meaning | NPU compiler relevance |
|---|---|---|
| One-Shot Bufferize | Analysis + rewrite pass for tensor -> memref IR | Main point where copies/allocations appear |
| DPS | Destination-passing style | Lets output buffers be reused safely |
| RaW conflict | Write happens before a later read of old value | Causes copy insertion or fresh allocation |
| alloc_tensor | Fresh tensor SSA chain anchor | Breaks alias conflicts but increases memory pressure |
| memory_space | Target-specific memory class in memref type/attribute | DRAM/SRAM/ACCUM mapping contract |
| lifetime | Definition-to-last-use interval | Basis for scratch buffer reuse |

## NPU bufferization checklist

1. Are high-level fusion and tiling finished before bufferization?
2. Are all tensor ops bufferizable or intentionally left at tensor/buffer boundary?
3. Are unexpected `memref.copy` or `bufferization.clone` operations absent?
4. Does function boundary bufferization match runtime tensor descriptor ABI?
5. Are memory spaces assigned in a way that survives lowering?
6. Is peak live SRAM below hardware budget?
7. Are double-buffer DMA and accumulator lifetimes non-overlapping where reuse is expected?

## Recommended pipeline sketch

```text
StableHLO/TOSA legalize
  -> Linalg/Tensor canonicalize
  -> Fuse/tile/vectorize on tensors
  -> Insert intentional alloc_tensor anchors only when needed
  -> One-Shot Bufferize
  -> Buffer deallocation / ownership
  -> Memory-space assignment
  -> Scratch buffer planning
  -> DMA/barrier/runtime lowering
```

## References

- MLIR Bufferization: https://mlir.llvm.org/docs/Bufferization/
- MLIR Bufferization Dialect: https://mlir.llvm.org/docs/Dialects/BufferizationOps/
- MLIR MemRef Dialect: https://mlir.llvm.org/docs/Dialects/MemRef/
- MLIR Builtin Dialect - MemRef type and memory space: https://mlir.llvm.org/docs/Dialects/Builtin/
- MLIR Tensor Dialect: https://mlir.llvm.org/docs/Dialects/TensorOps/
- MLIR Linalg Dialect: https://mlir.llvm.org/docs/Dialects/Linalg/
