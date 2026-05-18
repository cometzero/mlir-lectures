# MLIR NPU Lecture 06 - Runtime Import Notes

## Import-time metadata

Preserve or attach:

- model/subgraph name
- original framework op name/location
- static/dynamic dimension annotations
- layout intent if known
- quantization parameters and calibration origin
- numerical approximation policy for GELU/Softmax/LayerNorm

## Lowering boundary

```text
StableHLO/TOSA input
  -> canonicalize
  -> decompose composites
  -> audit target legal ops
  -> produce npu_graph candidate ops
```

Do not jump directly from StableHLO/TOSA to command buffers. The input IR is a semantic contract, not a hardware schedule.

## Minimum regression tests

1. LayerNorm with K aligned to vector lanes.
2. LayerNorm with K not aligned to vector lanes.
3. GELU exact vs approximate error.
4. MatMul + Bias + GELU fusion candidate.
5. Unsupported dynamic shape path.
6. Quantized activation path using rescale/table-style operations.
