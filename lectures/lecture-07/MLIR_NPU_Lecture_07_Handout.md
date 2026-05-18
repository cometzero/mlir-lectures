# MLIR NPU Lecture 07 - Linalg-on-Tensors

## 목표

- `linalg.generic`의 `indexing_maps`, `iterator_types`, `ins/outs`, region body를 읽는다.
- MatMul과 Conv2D NHWC/HWCF를 tensor-based Linalg IR로 표현한다.
- NPU compiler 관점에서 tiling, fusion, SRAM footprint, accumulator pressure를 분석한다.

## 핵심 개념

Linalg-on-Tensors는 high-level graph IR과 bufferization 사이의 structured compute layer입니다. Tensor value semantics를 유지하면서 loop/data mapping을 명시하므로, NPU backend는 실제 메모리 배치를 확정하기 전에 tile/fuse/vectorize 결정을 할 수 있습니다.

## linalg.generic 읽는 순서

1. `ins/outs`로 입력/결과 shape를 확인한다.
2. `indexing_maps`로 loop IV가 각 operand index로 어떻게 매핑되는지 확인한다.
3. `iterator_types`로 parallel/reduction loop를 구분한다.
4. region body에서 scalar compute와 `linalg.yield`를 확인한다.
5. NPU 관점에서 tile footprint, accumulator lifetime, epilogue fusion 가능성을 평가한다.

## MatMul pattern

```mlir
#matmul_trait = {
  indexing_maps = [
    affine_map<(m, n, k) -> (m, k)>,
    affine_map<(m, n, k) -> (k, n)>,
    affine_map<(m, n, k) -> (m, n)>
  ],
  iterator_types = ["parallel", "parallel", "reduction"]
}

func.func @matmul_generic(%A: tensor<8x16xf32>,
                          %B: tensor<16x32xf32>) -> tensor<8x32xf32> {
  %c0 = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<8x32xf32>
  %zero = linalg.fill ins(%c0 : f32)
                         outs(%init : tensor<8x32xf32>) -> tensor<8x32xf32>
  %C = linalg.generic #matmul_trait
    ins(%A, %B : tensor<8x16xf32>, tensor<16x32xf32>)
    outs(%zero : tensor<8x32xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %p = arith.mulf %a, %b : f32
    %s = arith.addf %c, %p : f32
    linalg.yield %s : f32
  } -> tensor<8x32xf32>
  return %C : tensor<8x32xf32>
}
```

## Conv2D NHWC/HWCF pattern

```mlir
#conv_trait = {
  indexing_maps = [
    affine_map<(n, oh, ow, f, kh, kw, c) -> (n, oh + kh, ow + kw, c)>,
    affine_map<(n, oh, ow, f, kh, kw, c) -> (kh, kw, c, f)>,
    affine_map<(n, oh, ow, f, kh, kw, c) -> (n, oh, ow, f)>
  ],
  iterator_types = ["parallel", "parallel", "parallel", "parallel",
                    "reduction", "reduction", "reduction"]
}

func.func @conv2d_nhwc_hwcf_generic(
    %input: tensor<1x30x30x16xf32>,
    %filter: tensor<3x3x16x32xf32>) -> tensor<1x28x28x32xf32> {
  %c0 = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<1x28x28x32xf32>
  %zero = linalg.fill ins(%c0 : f32)
                         outs(%init : tensor<1x28x28x32xf32>)
                         -> tensor<1x28x28x32xf32>
  %out = linalg.generic #conv_trait
    ins(%input, %filter : tensor<1x30x30x16xf32>, tensor<3x3x16x32xf32>)
    outs(%zero : tensor<1x28x28x32xf32>) {
  ^bb0(%x: f32, %w: f32, %acc: f32):
    %p = arith.mulf %x, %w : f32
    %s = arith.addf %acc, %p : f32
    linalg.yield %s : f32
  } -> tensor<1x28x28x32xf32>
  return %out : tensor<1x28x28x32xf32>
}
```

## NPU Audit Checklist

| 항목 | 질문 |
|---|---|
| Shape | static/dynamic shape가 backend ABI로 표현 가능한가? |
| Iterator | parallel/reduction 구분이 정확한가? |
| Access map | tile footprint와 layout 변환 비용을 계산할 수 있는가? |
| Fusion | bias/activation/requant를 epilogue에 붙일 수 있는가? |
| SRAM | input/filter/output/accumulator tile이 fast memory에 맞는가? |
| Lowering | Linalg를 유지할 단계와 NPU dialect로 낮출 단계를 구분했는가? |

## 참고자료

- MLIR Linalg Dialect: https://mlir.llvm.org/docs/Dialects/Linalg/
- Structured Linalg Tutorial: https://mlir.llvm.org/docs/Tutorials/transform/Ch0/
- MLIR Tensor Dialect: https://mlir.llvm.org/docs/Dialects/TensorOps/
