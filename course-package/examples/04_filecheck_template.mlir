// RUN: mlir-opt %s -canonicalize -cse | FileCheck %s
module {
  // CHECK-LABEL: func.func @matmul
  func.func @matmul(%A: tensor<16x16xf32>, %B: tensor<16x16xf32>, %C: tensor<16x16xf32>) -> tensor<16x16xf32> {
    // CHECK: linalg.matmul
    %0 = linalg.matmul ins(%A, %B : tensor<16x16xf32>, tensor<16x16xf32>)
                       outs(%C : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %0 : tensor<16x16xf32>
  }
}
