// Lecture 11 example: desired IR after NPU graph legalization.
module {
  func.func @matmul_bias_gelu(%a: tensor<128x256xf16>,
                              %b: tensor<256x128xf16>,
                              %bias: tensor<128xf16>) -> tensor<128x128xf16> {
    %0 = "npu_graph.matmul_epilogue"(%a, %b, %bias)
      {acc_type = i32, epilogue = "gelu_lut", k_tile = 32 : i64,
       layout = "A_row_B_col_C_row", approx_error = 0.001 : f64}
      : (tensor<128x256xf16>, tensor<256x128xf16>, tensor<128xf16>) -> tensor<128x128xf16>
    return %0 : tensor<128x128xf16>
  }
}
