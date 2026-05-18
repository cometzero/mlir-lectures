// Lecture 16 - INT4 weight packing pseudo IR.
// This is target-specific pseudo syntax for discussion.

func.func @int4_weight_only_matmul(
    %a_i8: tensor<1x128xi8>,
    %w_packed: memref<4096xi8, #npu.dram>,
    %scale: tensor<64x2xf32>,   // output_channel x K_group
    %mult: tensor<64x2xi32>,
    %shift: tensor<64x2xi8>) -> tensor<1x64xi8> {

  %y = "npu_kernel.qmatmul"(%a_i8, %w_packed, %scale, %mult, %shift) {
      lhs_type = "i8",
      rhs_type = "s4",
      rhs_packing = "two_per_byte_low_first",
      rhs_group_size = 64 : i64,
      rhs_scale_layout = "out_channel_major_then_k_group",
      acc_type = "i32",
      output_type = "i8",
      rounding = "nearest_even",
      saturation = "signed_i8"
    } : (tensor<1x128xi8>, memref<4096xi8, #npu.dram>, tensor<64x2xf32>, tensor<64x2xi32>, tensor<64x2xi8>)
      -> tensor<1x64xi8>
  return %y : tensor<1x64xi8>
}
