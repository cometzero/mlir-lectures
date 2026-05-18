# MLIR NPU Lecture 20 Quick Reference

## Capstone One-page Checklist
- Input IR contract: StableHLO/TOSA/Linalg, shape/dtype/layout policy
- Legal op set: dynamic legality, fallback, diagnostic message
- Pass pipeline: import, canonicalize, decompose, legalize, fuse, layout, quant, schedule, bufferize, lower, emit
- Schedule: tile size, fusion boundary, vector/tensorization, edge policy
- Memory: memref layout, memory_space, SRAM/ACCUM lifetime, DMA plan
- Dialect: op schema, verifier, assembly format, traits/interfaces
- Runtime: tensor descriptor, command packet, ABI version, capability query
- Validation: FileCheck, negative tests, golden tests, cost model

## Native Tile Example
`M=16, N=16, K=32`, INT8 x INT8 -> INT32 accumulator.

## Minimum Positive Test
`MatMul + Bias + ReLU` lowers to one fused NPU matmul epilogue and one store.

## Minimum Negative Test
Invalid tile, unsupported dtype/layout, or missing barrier emits a clear diagnostic.
