// Lecture 07 example: MatMul + Bias + ReLU fusion candidate sketch.
// Use this to discuss producer/consumer fusion and epilogue placement.

func.func @matmul_bias_relu_candidate(%A: tensor<8x16xf32>,
                                      %B: tensor<16x32xf32>,
                                      %bias: tensor<32xf32>) -> tensor<8x32xf32> {
  %c0 = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<8x32xf32>
  %zero = linalg.fill ins(%c0 : f32) outs(%init : tensor<8x32xf32>) -> tensor<8x32xf32>
  %mm = linalg.matmul ins(%A, %B : tensor<8x16xf32>, tensor<16x32xf32>)
                     outs(%zero : tensor<8x32xf32>) -> tensor<8x32xf32>

  // TODO: express bias add + relu as linalg.generic with broadcast map.
  // NPU question: can this become a matmul epilogue in npu_kernel.matmul?
  return %mm : tensor<8x32xf32>
}
