// RUN: mlir-opt %s --pass-pipeline='builtin.module(func.func(cse,canonicalize))' | FileCheck %s

module {
  func.func @matmul_bias_relu(%A: tensor<128x64xf32>, %B: tensor<64x128xf32>, %bias: tensor<128xf32>) -> tensor<128x128xf32> {
    %init = tensor.empty() : tensor<128x128xf32>
    %C = linalg.matmul ins(%A, %B : tensor<128x64xf32>, tensor<64x128xf32>)
                       outs(%init : tensor<128x128xf32>) -> tensor<128x128xf32>
    // Pseudo epilogue. Later lectures will replace this with structured linalg.generic.
    %zero = arith.constant 0.000000e+00 : f32
    %out = "npu_graph.pseudo_bias_relu"(%C, %bias, %zero) : (tensor<128x128xf32>, tensor<128xf32>, f32) -> tensor<128x128xf32>
    return %out : tensor<128x128xf32>
  }
}

// CHECK-LABEL: func.func @matmul_bias_relu
// CHECK: linalg.matmul
