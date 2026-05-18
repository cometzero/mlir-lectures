# MLIR NPU Compiler - Lecture 13 Handout

**Topic:** MemRef, Layout, Memory Space  
**Goal:** DRAM/SRAM/Accumulator memory mapping for NPU compiler  
**Generated:** 2026-05-07

## Key message

12강에서 tensor를 memref로 낮췄다면, 13강은 memref를 실제 NPU storage contract로 읽는 단계입니다. 이제 buffer는 단순한 shape가 아니라 layout, memory space, descriptor ABI를 함께 가집니다.

```mlir
memref<64x128xf16, strided<[160, 1], offset: ?>, #npu.mem<"sram", bank=0, align=64>>
```

## What to read in a memref

| Field | Meaning | NPU compiler question |
|---|---|---|
| Shape/rank | logical index space | tile size and loop bounds? |
| Element type | f16/i8/i32 | bandwidth, vector lanes, accumulator width? |
| Layout | affine map or strides | DMA stride, bank conflict, address generator? |
| Memory space | target-specific attr | DRAM/SRAM/ACCUM placement and ABI? |

## Memory space convention for this lecture

| Space | Meaning | Typical object |
|---|---|---|
| 0 / DRAM | external memory | inputs, outputs, large activations, weights |
| 1 / SRAM | scratchpad | A/B/C tile buffers, ping-pong buffers |
| 2 / ACCUM | accumulator | i32/f32 matmul/reduction result |
| 3 / CONST | constant memory | weights, LUTs |

## MatMul mapping

```text
A[MxK] DRAM -> A_tile[Mt x Kt] SRAM bank0
B[KxN] DRAM -> B_tile[Kt x Nt] SRAM bank1
C_acc[Mt x Nt] ACCUM -> C_stage SRAM -> C[MxN] DRAM
```

## Verification checklist

- `memref.load/store` index count equals rank.
- `subview` offsets/sizes/strides are in-bounds or runtime-guarded.
- SRAM buffers fit capacity and lifetime schedule.
- ACCUM buffers do not escape to host-visible ABI unless supported.
- `memory_space_cast` is not used as a replacement for DMA copy.
- Runtime descriptor preserves dynamic sizes/strides/offset.

## References
- MLIR MemRef Dialect: https://mlir.llvm.org/docs/Dialects/MemRef/
- MLIR Builtin Dialect - MemRef type: https://mlir.llvm.org/docs/Dialects/Builtin/
- MLIR Data Layout Modeling: https://mlir.llvm.org/docs/DataLayout/
- MLIR FAQ - MemRef is not only a pointer: https://mlir.llvm.org/getting_started/Faq/
- MLIR Passes: https://mlir.llvm.org/docs/Passes/
