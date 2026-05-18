// RUN: mlir-opt %s --npu-fuse-epilogue | FileCheck %s

func.func @matmul_bias_relu(%A: tensor<128x256xf32>, %B: tensor<256x512xf32>,
                            %bias: tensor<512xf32>) -> tensor<128x512xf32> {
  // CHECK-LABEL: func.func @matmul_bias_relu
  // CHECK: "npu_graph.matmul_epilogue"
  // CHECK-SAME: activation = "relu"
  // CHECK-NOT: linalg.generic
  %y = "test.matmul_add_relu_pattern"(%A, %B, %bias)
      : (tensor<128x256xf32>, tensor<256x512xf32>, tensor<512xf32>) -> tensor<128x512xf32>
  return %y : tensor<128x512xf32>
}
