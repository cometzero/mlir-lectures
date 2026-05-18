# Lecture 15 Quick Reference - Transform Dialect

## Core concepts
- `payload IR`: IR being transformed.
- `transform IR`: IR controlling the transformation.
- `operation handle`: associated with payload operations.
- `value handle`: associated with payload SSA values.
- `parameter`: associated with compile-time/runtime-known attributes, often tile sizes.

## Common ops to recognize
- `transform.named_sequence`
- `transform.structured.match`
- `transform.structured.tile_using_for`
- `transform.structured.tile_using_forall`
- `transform.structured.fuse_into_containing_op`
- `transform.structured.vectorize`
- `transform.apply_patterns`

## NPU mapping
- Match: select matmul/conv/epilogue payload ops.
- Tile: map M/N/K or H/W/C/K to SRAM and native array shapes.
- Fuse: keep epilogue or producer slice local to tile.
- Vectorize: create target-neutral vector IR.
- Tensorize: lower `vector.contract` to `npu_kernel.matmul` or equivalent.

## Debug commands
```bash
mlir-opt input.mlir --transform-interpreter --mlir-print-ir-after-change
mlir-opt input.mlir --transform-dialect-check-uses
mlir-opt input.mlir --transform-infer-effects
```
