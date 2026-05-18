// Lecture 17: layout propagation pseudo IR.
// The compiler propagates physical layout metadata through elementwise ops and
// materializes repack only at incompatible boundaries.

func.func @layout_propagation(%a: tensor<128x4096xf16>, %w: tensor<4096x4096xf16>, %b: tensor<4096xf16>) -> tensor<128x4096xf16> {
  %w_packed = "npu_graph.prepack_weight"(%w) {
    src_layout = "row_major", dst_layout = "KxN_blocked_32x16"
  } : (tensor<4096x4096xf16>) -> tensor<4096x4096xf16, #npu.layout<"KxN_blocked_32x16">>

  %out = "npu_graph.matmul_epilogue"(%a, %w_packed, %b) {
    out_layout = "MxN_blocked_16x16", epilogue = "bias_relu"
  } : (tensor<128x4096xf16>, tensor<4096x4096xf16, #npu.layout<"KxN_blocked_32x16">>, tensor<4096xf16>)
    -> tensor<128x4096xf16, #npu.layout<"MxN_blocked_16x16">>

  // This op supports blocked layout. No materialization needed.
  %scaled = "npu_graph.mul_scalar"(%out) {scale = 0.125 : f32, layout_passthrough = true}
    : (tensor<128x4096xf16, #npu.layout<"MxN_blocked_16x16">>) -> tensor<128x4096xf16, #npu.layout<"MxN_blocked_16x16">>

  // This fallback op supports only row-major. Insert one explicit materialization point.
  %row = "npu_graph.layout_materialize"(%scaled) {from = "MxN_blocked_16x16", to = "row_major", reason = "fallback_row_major_only"}
    : (tensor<128x4096xf16, #npu.layout<"MxN_blocked_16x16">>) -> tensor<128x4096xf16>

  return %row : tensor<128x4096xf16>
}
