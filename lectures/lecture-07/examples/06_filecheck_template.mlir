// RUN: mlir-opt %s --canonicalize --cse | FileCheck %s

// CHECK-LABEL: func.func @matmul_generic
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: linalg.yield

#matmul_trait = {
  indexing_maps = [
    affine_map<(m, n, k) -> (m, k)>,
    affine_map<(m, n, k) -> (k, n)>,
    affine_map<(m, n, k) -> (m, n)>
  ],
  iterator_types = ["parallel", "parallel", "reduction"]
}

func.func @matmul_generic(%A: tensor<8x16xf32>, %B: tensor<16x32xf32>) -> tensor<8x32xf32> {
  %c0 = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<8x32xf32>
  %zero = linalg.fill ins(%c0 : f32) outs(%init : tensor<8x32xf32>) -> tensor<8x32xf32>
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
