// Lecture 07 example: linalg.generic Conv2D NHWC/HWCF on tensors

#conv_trait = {
  indexing_maps = [
    affine_map<(n, oh, ow, f, kh, kw, c) -> (n, oh + kh, ow + kw, c)>,
    affine_map<(n, oh, ow, f, kh, kw, c) -> (kh, kw, c, f)>,
    affine_map<(n, oh, ow, f, kh, kw, c) -> (n, oh, ow, f)>
  ],
  iterator_types = ["parallel", "parallel", "parallel", "parallel",
                    "reduction", "reduction", "reduction"]
}

func.func @conv2d_nhwc_hwcf_generic(
    %input: tensor<1x30x30x16xf32>,
    %filter: tensor<3x3x16x32xf32>) -> tensor<1x28x28x32xf32> {
  %c0 = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<1x28x28x32xf32>
  %zero = linalg.fill ins(%c0 : f32)
                         outs(%init : tensor<1x28x28x32xf32>)
                         -> tensor<1x28x28x32xf32>
  %out = linalg.generic #conv_trait
    ins(%input, %filter : tensor<1x30x30x16xf32>, tensor<3x3x16x32xf32>)
    outs(%zero : tensor<1x28x28x32xf32>) {
  ^bb0(%x: f32, %w: f32, %acc: f32):
    %p = arith.mulf %x, %w : f32
    %s = arith.addf %acc, %p : f32
    linalg.yield %s : f32
  } -> tensor<1x28x28x32xf32>
  return %out : tensor<1x28x28x32xf32>
}
