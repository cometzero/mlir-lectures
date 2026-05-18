// RUN: mlir-opt %s --npu-legalize-quant --npu-materialize-requant | FileCheck %s

func.func @matmul_bias_relu_qdq(%a: tensor<1x128xf32>, %w: tensor<128x64xf32>, %b: tensor<64xf32>) -> tensor<1x64xf32> {
  // Pseudo QDQ graph.
  %qa = quant.qcast %a : tensor<1x128xf32> to tensor<1x128x!quant.uniform<i8:f32, 0.03125:-3>>
  %qw = quant.qcast %w : tensor<128x64xf32> to tensor<128x64x!quant.uniform<i8:f32:1, {0.015, 0.017}>>
  %dq_a = quant.dcast %qa : tensor<1x128x!quant.uniform<i8:f32, 0.03125:-3>> to tensor<1x128xf32>
  %dq_w = quant.dcast %qw : tensor<128x64x!quant.uniform<i8:f32:1, {0.015, 0.017}>> to tensor<128x64xf32>
  %0 = linalg.matmul ins(%dq_a, %dq_w : tensor<1x128xf32>, tensor<128x64xf32>) outs(%b : tensor<64xf32>) -> tensor<1x64xf32>
  return %0 : tensor<1x64xf32>
}

// CHECK-LABEL: func.func @matmul_bias_relu_qdq
// CHECK: "npu_kernel.qmatmul"
// CHECK-SAME: acc_type = "i32"
// CHECK-SAME: rounding = "nearest_even"
// CHECK: tosa.rescale
// CHECK-SAME: per_channel = true
// CHECK-NOT: quant.dcast
