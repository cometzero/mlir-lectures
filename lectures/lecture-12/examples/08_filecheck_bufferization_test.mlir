// RUN: mlir-opt %s -one-shot-bufferize -canonicalize | FileCheck %s

// CHECK-LABEL: func.func @no_unexpected_copy
// CHECK-NOT: memref.copy
func.func @no_unexpected_copy(%A: tensor<16x16xf32>, %B: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %C0 = tensor.empty() : tensor<16x16xf32>
  %C = linalg.matmul ins(%A, %B : tensor<16x16xf32>, tensor<16x16xf32>)
                    outs(%C0 : tensor<16x16xf32>) -> tensor<16x16xf32>
  return %C : tensor<16x16xf32>
}
