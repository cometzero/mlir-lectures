# MLIR NPU Lecture 14 Quick Reference

## Vector Type

```text
vector<16x32xi8>  -> 2-D vector, often interpreted as MxK tile
vector<32x16xi8>  -> 2-D vector, often interpreted as KxN tile
vector<16x16xi32> -> accumulator/result tile
```

## MatMul Contract Trait

```mlir
#matmul_trait = {
  indexing_maps = [
    affine_map<(m, n, k) -> (m, k)>,
    affine_map<(m, n, k) -> (k, n)>,
    affine_map<(m, n, k) -> (m, n)>
  ],
  iterator_types = ["parallel", "parallel", "reduction"]
}
```

## Tensorization Rule of Thumb

```text
Tensorize iff:
  vector.contract is semantically native matmul
  + M/N/K shape matches target tile
  + lhs/rhs/acc dtype are legal
  + layout/permutation maps are supported
  + mask/remainder policy is legal
  + operands live in legal memory spaces
```

## FileCheck Template

```mlir
// RUN: mlir-opt %s --npu-vector-contract-tensorize | FileCheck %s
// CHECK-LABEL: func.func @m16n16k32_i8
// CHECK: npu_kernel.matmul
// CHECK-SAME: m = 16
// CHECK-SAME: n = 16
// CHECK-SAME: k = 32
// CHECK-NOT: vector.contract
```

## NPU Native Tile Example

```text
M=16, N=16, K=32
A: vector<16x32xi8>
B: vector<32x16xi8>
C/Acc: vector<16x16xi32>
```
