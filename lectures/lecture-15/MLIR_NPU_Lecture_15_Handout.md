# MLIR NPU Compiler - Lecture 15 Handout

## Topic
Transform Dialect: `tile/fuse/vectorize` schedule IR for NPU compiler.

## Mental Model
- **Payload IR**: the IR being transformed, e.g., `linalg.matmul`.
- **Transform IR**: a schedule program written as MLIR that selects payload objects and applies transformations.
- **Handle**: a Transform SSA value associated with payload operations or values.
- **Parameter**: a schedule value such as tile size or vector size.

## NPU Schedule Pattern
```text
match linalg.matmul
  -> tile M/N/K according to native tile shape
  -> fuse epilogue when accumulator locality is beneficial
  -> vectorize tiled op
  -> tensorize vector.contract to npu_kernel.matmul
  -> bufferize and assign memory spaces
```

## Rule of Thumb
Do high-level tile/fuse/vectorize on tensor/Linalg IR before bufferization. After bufferization, layout and memory-space constraints become much harder to change.

## Native Tile Example
- M tile: 16
- N tile: 16
- K tile: 32
- Input: INT8 x INT8
- Accumulator: INT32
- Epilogue: bias + requant + relu

## Review Checklist
See `examples/08_schedule_review_checklist.md`.
