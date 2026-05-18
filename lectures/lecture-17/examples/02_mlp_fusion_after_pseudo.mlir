// Lecture 17 example: after graph-level fusion and layout propagation.
// Custom npu_graph ops are pseudo and are used to express compiler intent.

func.func @mlp_gelu_fused(%x: tensor<128x4096xf16>,
                          %w1_packed: tensor<4096x16384xf16, #npu.layout<"KxN_blocked_32x16">>,
                          %b1: tensor<16384xf16>,
                          %w2_packed: tensor<16384x4096xf16, #npu.layout<"KxN_blocked_32x16">>,
                          %b2: tensor<4096xf16>) -> tensor<128x4096xf16> {
  %h = "npu_graph.matmul_epilogue"(%x, %w1_packed, %b1) {
      epilogue = "bias_gelu_tanh",
      lhs_layout = "MxK_row",
      rhs_layout = "KxN_blocked_32x16",
      out_layout = "MxN_blocked_16x16",
      accumulator = "fp32",
      fusion_id = "mlp.gemm1.bias.gelu"
    } : (tensor<128x4096xf16>, tensor<4096x16384xf16, #npu.layout<"KxN_blocked_32x16">>, tensor<16384xf16>)
      -> tensor<128x16384xf16, #npu.layout<"MxN_blocked_16x16">>

  %y = "npu_graph.matmul_epilogue"(%h, %w2_packed, %b2) {
      epilogue = "bias",
      lhs_layout = "MxK_blocked_16x16",
      rhs_layout = "KxN_blocked_32x16",
      out_layout = "MxN_row",
      accumulator = "fp32",
      fusion_id = "mlp.gemm2.bias"
    } : (tensor<128x16384xf16, #npu.layout<"MxN_blocked_16x16">>, tensor<16384x4096xf16, #npu.layout<"KxN_blocked_32x16">>, tensor<4096xf16>)
      -> tensor<128x4096xf16>
  return %y : tensor<128x4096xf16>
}
