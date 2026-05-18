# Lecture 17 Quick Reference

## Fusion legality fields

```text
shape_static | rank | dtype | accumulator | quant_axis | broadcast
layout_in | layout_out | single_use | sram_footprint | materialization_cost
```

## Common fusion groups

```text
GEMM + Bias
GEMM + Bias + ReLU/GELU/SiLU
GEMM + Requant + Clamp
Elementwise chain
SwiGLU: gate/up + SiLU + Mul
Layout-preserving reshape/view + consumer
```

## Layout propagation rule of thumb

```text
Propagate through: elementwise, view-like reshape, tile-local activation
Stop at: reduction with incompatible axis, row-major-only fallback, function ABI, CPU fallback
Materialize explicitly with: npu_graph.layout_materialize
```

## FileCheck anchors

```mlir
// CHECK: "npu_graph.matmul_epilogue"
// CHECK-SAME: epilogue = "bias_gelu_tanh"
// CHECK-SAME: out_layout = "MxN_blocked_16x16"
// CHECK-NOT: "test.bias"
// CHECK-NOT: "test.gelu"
```
