// Lecture 16 - INT8 MatMul + requant pseudo MLIR.
// Demonstrates the numeric contract expected by an NPU backend.

func.func @int8_matmul_requant(
    %a: tensor<1x128xi8>,
    %w: tensor<128x64xi8>,
    %bias: tensor<64xi32>,
    %mult: tensor<64xi32>,
    %shift: tensor<64xi8>) -> tensor<1x64xi8> {
  // Pseudo NPU op: output is i32 accumulator.
  %acc = "npu_kernel.qmatmul"(%a, %w, %bias) {
      lhs_zp = -3 : i64,
      rhs_zp = 0 : i64,
      lhs_type = "i8",
      rhs_type = "i8",
      acc_type = "i32",
      rhs_scale_granularity = "per_output_channel"
    } : (tensor<1x128xi8>, tensor<128x64xi8>, tensor<64xi32>) -> tensor<1x64xi32>

  // TOSA-style bridge for integer requantization.
  %out_zp = arith.constant dense<5> : tensor<i32>
  %in_zp = arith.constant dense<0> : tensor<i32>
  %y = tosa.rescale %acc, %mult, %shift, %in_zp, %out_zp {
      scale32 = true,
      rounding_mode = #tosa.rounding_mode<SINGLE_ROUND>,
      per_channel = true,
      input_unsigned = false,
      output_unsigned = false
    } : (tensor<1x64xi32>, tensor<64xi32>, tensor<64xi8>, tensor<i32>, tensor<i32>) -> tensor<1x64xi8>
  return %y : tensor<1x64xi8>
}
