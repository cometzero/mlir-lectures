// RUN: not mlir-opt %s --npu-verify-quant 2>&1 | FileCheck %s

// CHECK: error: per-channel scale count does not match output channel dimension
func.func @bad_scale_count(%w: tensor<128x64xf32>) {
  %qw = quant.qcast %w : tensor<128x64xf32>
      to tensor<128x64x!quant.uniform<i8:f32:1, {0.015, 0.017}>>
  return
}

// -----

// CHECK: error: INT4 group size does not divide K dimension
func.func @bad_int4_group(%a: tensor<1x130xi8>, %w: memref<4160xi8>) {
  %y = "npu_kernel.qmatmul"(%a, %w) {
      rhs_type = "s4",
      rhs_group_size = 64 : i64
    } : (tensor<1x130xi8>, memref<4160xi8>) -> tensor<1x64xi8>
  return
}
