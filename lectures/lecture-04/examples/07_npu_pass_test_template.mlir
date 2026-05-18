// Lecture 04 - Example 07: pseudo NPU pass regression template.
// This is a template for your future custom tool, e.g. npu-opt.
// RUN: npu-opt %s --pass-pipeline='builtin.module(func.func(npu-fuse,npu-tile{tile-m=16 tile-n=16 tile-k=32}))' | FileCheck %s

// CHECK-LABEL: func.func @matmul_bias_relu
// CHECK: npu.matmul
// CHECK-SAME: tile_m = 16
// CHECK-SAME: tile_n = 16
// CHECK-SAME: tile_k = 32
// CHECK: npu.relu
// CHECK-NOT: stablehlo.dot_general
func.func @matmul_bias_relu(%a: tensor<128x256xf32>, %b: tensor<256x64xf32>, %bias: tensor<64xf32>) -> tensor<128x64xf32> {
  // Pseudo body: replace this with StableHLO/TOSA/Linalg input in later lectures.
  %0 = "stablehlo.dot_general"(%a, %b) : (tensor<128x256xf32>, tensor<256x64xf32>) -> tensor<128x64xf32>
  %1 = "stablehlo.add"(%0, %bias) : (tensor<128x64xf32>, tensor<64xf32>) -> tensor<128x64xf32>
  %2 = "stablehlo.maximum"(%1, %1) : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
  return %2 : tensor<128x64xf32>
}
