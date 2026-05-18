# MLIR NPU Lecture 12 Quick Reference

## Commands

```bash
mlir-opt input.mlir -one-shot-bufferize -canonicalize
mlir-opt input.mlir -one-shot-bufferize='bufferize-function-boundaries=true'
mlir-opt input.mlir -one-shot-bufferize -debug-only=one-shot-bufferize
```

## IR markers to audit

- `tensor.empty`: possible destination allocation seed
- `bufferization.alloc_tensor`: explicit fresh allocation anchor
- `bufferization.to_tensor` / `bufferization.to_buffer`: tensor/buffer boundary
- `memref.alloc`: physical allocation
- `memref.copy` or `bufferization.clone`: possible performance issue; verify correctness reason
- `memref.dealloc`: ownership/deallocation correctness
- `memref.subview`: aliasing region
- `memref.memory_space_cast`: target memory-space boundary

## In-place decision cheat sheet

| Check | In-place likely? | Notes |
|---|---|---|
| DPS destination exists | yes | especially linalg outs |
| Later read of old destination | no | RaW conflict |
| Shape/layout compatible | yes | otherwise cast/copy may appear |
| Unknown tensor op at boundary | maybe no | to_buffer/to_tensor inserted |
| alloc_tensor used | fresh buffer | no alias with previous chain |

## NPU memory-space sketch

```text
0 = DRAM / external memory
1 = SRAM / scratchpad
2 = ACCUM / partial-sum buffer
3 = constant / weights cache
```

Use custom dialect attributes in production rather than relying only on integer spaces.
