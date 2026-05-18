// RUN: mlir-opt %s --pass-pipeline='builtin.module(func.func(npu-fuse-matmul-epilogue))' | FileCheck %s

module {
  func.func @case(%A: tensor<128x64xf32>, %B: tensor<64x128xf32>, %bias: tensor<128xf32>) -> tensor<128x128xf32> {
    // This is pseudo IR for test design. Replace with real linalg/arithmetic in implementation.
    %0 = "test.matmul"(%A, %B) : (tensor<128x64xf32>, tensor<64x128xf32>) -> tensor<128x128xf32>
    %1 = "test.bias"(%0, %bias) : (tensor<128x128xf32>, tensor<128xf32>) -> tensor<128x128xf32>
    %2 = "test.relu"(%1) : (tensor<128x128xf32>) -> tensor<128x128xf32>
    return %2 : tensor<128x128xf32>
  }
}

// CHECK-LABEL: func.func @case
// CHECK: "npu_graph.matmul"
// CHECK-SAME: epilogue = "relu"
// CHECK-NOT: "test.bias"
// CHECK-NOT: "test.relu"
