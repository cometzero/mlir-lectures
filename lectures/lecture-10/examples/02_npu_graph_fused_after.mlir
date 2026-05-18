// Lecture 10 expected pseudo IR after fusion.
// npu_graph.matmul is a teaching dialect used to show the target contract.

func.func @matmul_bias_relu(%A: tensor<128x256xf16>,
                            %B: tensor<256x128xf16>,
                            %bias: tensor<128xf16>) -> tensor<128x128xf16> {
  %out = "npu_graph.matmul"(%A, %B, %bias) {
      epilogue = ["bias", "relu"],
      accumulator = "f32",
      output_layout = "row_major",
      tile_m = 32 : i64,
      tile_n = 64 : i64,
      tile_k = 64 : i64
    } : (tensor<128x256xf16>, tensor<256x128xf16>, tensor<128xf16>)
        -> tensor<128x128xf16>
  return %out : tensor<128x128xf16>
}
