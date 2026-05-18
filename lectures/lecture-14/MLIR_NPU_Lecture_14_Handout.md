# Lecture 14 - Vector Dialect and Tensorization

## Goal

이번 강의의 목표는 MLIR `vector` dialect를 NPU compiler의 tile compute IR로 해석하고, `vector.contract`를 native NPU matmul/tensor instruction으로 tensorization하는 기준을 세우는 것입니다.

## Key Mental Model

```text
Linalg/Tensor
  -> vector.transfer_read / vector.contract / vector.transfer_write
  -> npu_kernel.matmul / npu_kernel.dma / npu_kernel.barrier
  -> command buffer / runtime ABI
```

`vector.contract`는 target-neutral contraction입니다. NPU backend는 이를 target-specific op로 바꾸기 전에 shape, dtype, accumulator, layout, mask, memory-space를 모두 검증해야 합니다.

## Core Ops

| Op | Role | NPU View |
|---|---|---|
| `vector.transfer_read` | tile/slice를 SSA vector로 읽기 | DMA/load tile boundary |
| `vector.transfer_write` | SSA vector를 tile/slice로 쓰기 | store/write-back boundary |
| `vector.contract` | matmul/dot/reduction contraction | native tensor op 후보 |
| `vector.transpose` | vector dimension permutation | layout repair 또는 cost |
| `vector.multi_reduction` | multi-dim reduction | reduction engine 또는 fallback |
| `vector.mask` / `vector.create_mask` | predication | edge tile handling |

## Tensorization Checklist

1. `iterator_types`가 matmul 형태인가? `parallel, parallel, reduction`
2. `indexing_maps`가 A(m,k), B(k,n), C(m,n)인가?
3. vector shape가 native M/N/K tile과 일치하는가?
4. lhs/rhs/acc/result dtype이 hardware policy와 일치하는가?
5. mask/remainder를 native NPU가 지원하는가?
6. layout/permutation이 native datapath와 맞는가?
7. operands가 correct memory space에 있고 alignment가 맞는가?
8. tensorization 후 FileCheck로 `vector.contract`가 제거되었는가?

## Example

```mlir
#matmul_trait = {
  indexing_maps = [
    affine_map<(m, n, k) -> (m, k)>,
    affine_map<(m, n, k) -> (k, n)>,
    affine_map<(m, n, k) -> (m, n)>
  ],
  iterator_types = ["parallel", "parallel", "reduction"]
}
%c = vector.contract #matmul_trait %a, %b, %acc
  : vector<16x32xi8>, vector<32x16xi8> into vector<16x16xi32>
```

Possible NPU lowering:

```mlir
%c2 = npu_kernel.matmul %a, %b, %acc
  {m = 16 : i64, n = 16 : i64, k = 32 : i64,
   lhs_layout = "mk", rhs_layout = "kn", acc_type = "i32"}
  : vector<16x32xi8>, vector<32x16xi8>, vector<16x16xi32>
    -> vector<16x16xi32>
```

## Common Mistakes

- `vector.contract`만 보고 무조건 NPU op로 바꾸는 것
- `permutation_map`과 layout을 무시하는 것
- edge tile mask를 correctness 문제가 아닌 performance 문제로만 보는 것
- accumulator type과 requant/epilogue policy를 분리하지 않는 것
- unsupported vector shape를 scalar lowering으로 보내고 성능 회귀를 놓치는 것
