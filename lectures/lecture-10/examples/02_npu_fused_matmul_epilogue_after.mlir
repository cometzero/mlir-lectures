// Lecture 10 - expected fused pseudo IR.
// npu_graph is a teaching dialect name used to show the target contract.

func.func @matmul_bias_relu(
    %A : tensor<128x256xf32>,
    %B : tensor<256x512xf32>,
    %bias : tensor<512xf32>) -> tensor<128x512xf32> {
  %y = "npu_graph.matmul_epilogue"(%A, %B, %bias) {
      bias = true,
      activation = "relu",
      lhs_layout = "row_major",
      rhs_layout = "row_major",
      out_layout = "row_major",
      accumulator = "f32"
    } : (tensor<128x256xf32>, tensor<256x512xf32>, tensor<512xf32>) -> tensor<128x512xf32>
  return %y : tensor<128x512xf32>
}
