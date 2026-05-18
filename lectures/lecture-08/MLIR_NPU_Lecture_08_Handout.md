# MLIR NPU Compiler - Lecture 08 Handout

## Topic
SCF/Affine Loop: M/N/K tile loop and DMA scheduling for NPU compiler.

## Mental model
`Linalg` describes what structured compute does. `SCF/Affine` begins to expose how the compiler will iterate, tile, load, compute, and store.

## Key concepts
- SCF is useful for dynamic bounds, runtime condition, and scheduling skeletons.
- Affine is useful for static loop/access analysis and polyhedral-style transforms.
- NPU compiler loop order controls accumulator reuse, input tile streaming, and DMA overlap.
- Tile size is a hardware contract: SRAM, bank, alignment, accumulator, MAC array, and DMA burst.

## SRAM footprint
```text
A_tile = TM * TK * input_bytes
B_tile = TK * TN * input_bytes
Acc_tile = TM * TN * acc_bytes
WorkingSet ~= 2*(A_tile+B_tile) + Acc_tile    # double-buffered inputs
```

Example: `TM=32, TN=64, TK=64`, int8 input, int32 accumulator:
```text
A = 2 KB, B = 4 KB, Acc = 8 KB, total ~= 20 KB + overhead
```

## Core exercise
Design an M/N/K tiled matmul loop nest and DMA ordering for `MatMul(128x256 x 256x128)` with `TM=32`, `TN=64`, `TK=64`.

## References
- https://mlir.llvm.org/docs/Dialects/SCFDialect/
- https://mlir.llvm.org/docs/Dialects/Affine/
- https://mlir.llvm.org/docs/Dialects/MemRef/
- https://mlir.llvm.org/docs/Dialects/Linalg/
