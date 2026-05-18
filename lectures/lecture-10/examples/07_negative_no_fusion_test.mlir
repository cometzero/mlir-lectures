// RUN: mlir-opt %s --npu-fuse-epilogue | FileCheck %s

func.func @shared_matmul_result_must_not_fuse(
    %A: tensor<128x256xf32>, %B: tensor<256x512xf32>, %bias: tensor<512xf32>)
    -> (tensor<128x512xf32>, tensor<128x512xf32>) {
  // CHECK-LABEL: func.func @shared_matmul_result_must_not_fuse
  // CHECK-NOT: "npu_graph.matmul_epilogue"
  // CHECK: "test.matmul_add_relu_shared"
  %mm, %relu = "test.matmul_add_relu_shared"(%A, %B, %bias)
      : (tensor<128x256xf32>, tensor<256x512xf32>, tensor<512xf32>)
        -> (tensor<128x512xf32>, tensor<128x512xf32>)
  return %mm, %relu : tensor<128x512xf32>, tensor<128x512xf32>
}
