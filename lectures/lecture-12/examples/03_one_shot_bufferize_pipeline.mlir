// RUN: mlir-opt %s -one-shot-bufferize="bufferize-function-boundaries=true" -canonicalize | FileCheck %s
// CHECK-LABEL: func.func @expected_inplace
// CHECK-NOT: memref.copy

func.func @expected_inplace(%A: tensor<32x32xf32>, %B: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %C0 = tensor.empty() : tensor<32x32xf32>
  %C = linalg.matmul ins(%A, %B : tensor<32x32xf32>, tensor<32x32xf32>)
                    outs(%C0 : tensor<32x32xf32>) -> tensor<32x32xf32>
  return %C : tensor<32x32xf32>
}
