// Lecture 07 example: named Linalg ops

func.func @named_ops(%A: tensor<8x16xf32>,
                     %B: tensor<16x32xf32>,
                     %input: tensor<1x30x30x16xf32>,
                     %filter: tensor<3x3x16x32xf32>)
    -> (tensor<8x32xf32>, tensor<1x28x28x32xf32>) {
  %c0 = arith.constant 0.0 : f32
  %mm_init = tensor.empty() : tensor<8x32xf32>
  %mm_zero = linalg.fill ins(%c0 : f32)
                            outs(%mm_init : tensor<8x32xf32>) -> tensor<8x32xf32>
  %mm = linalg.matmul ins(%A, %B : tensor<8x16xf32>, tensor<16x32xf32>)
                      outs(%mm_zero : tensor<8x32xf32>) -> tensor<8x32xf32>

  %cv_init = tensor.empty() : tensor<1x28x28x32xf32>
  %cv_zero = linalg.fill ins(%c0 : f32)
                            outs(%cv_init : tensor<1x28x28x32xf32>)
                            -> tensor<1x28x28x32xf32>
  %cv = linalg.conv_2d_nhwc_hwcf
          ins(%input, %filter : tensor<1x30x30x16xf32>, tensor<3x3x16x32xf32>)
          outs(%cv_zero : tensor<1x28x28x32xf32>) -> tensor<1x28x28x32xf32>
  return %mm, %cv : tensor<8x32xf32>, tensor<1x28x28x32xf32>
}
