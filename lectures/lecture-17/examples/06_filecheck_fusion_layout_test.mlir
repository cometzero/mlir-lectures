// RUN: mlir-opt %s --npu-fuse-mlp-epilogue --npu-propagate-layout \
// RUN:   | FileCheck %s

func.func @positive(%x: tensor<128x4096xf16>, %w: tensor<4096x16384xf16>, %b: tensor<16384xf16>) -> tensor<128x16384xf16> {
  // Pseudo input: matmul -> bias -> gelu.
  %0 = "test.matmul"(%x, %w) : (tensor<128x4096xf16>, tensor<4096x16384xf16>) -> tensor<128x16384xf16>
  %1 = "test.bias"(%0, %b) : (tensor<128x16384xf16>, tensor<16384xf16>) -> tensor<128x16384xf16>
  %2 = "test.gelu"(%1) : (tensor<128x16384xf16>) -> tensor<128x16384xf16>
  return %2 : tensor<128x16384xf16>
}

// CHECK-LABEL: func.func @positive
// CHECK: "npu_graph.matmul_epilogue"
// CHECK-SAME: epilogue = "bias_gelu_tanh"
// CHECK-SAME: out_layout = "MxN_blocked_16x16"
// CHECK-NOT: "test.bias"
// CHECK-NOT: "test.gelu"
