// MatMul + Bias + ReLU in tensor land.
// Intended input to a fusion + bufferization exercise.

#map2 = affine_map<(i, j) -> (i, j)>
#bias = affine_map<(i, j) -> (j)>

func.func @matmul_bias_relu(%A: tensor<128x256xf32>,
                            %B: tensor<256x128xf32>,
                            %bias: tensor<128xf32>) -> tensor<128x128xf32> {
  %zero = arith.constant 0.0 : f32
  %C0 = tensor.empty() : tensor<128x128xf32>
  %C1 = linalg.matmul ins(%A, %B : tensor<128x256xf32>, tensor<256x128xf32>)
                     outs(%C0 : tensor<128x128xf32>) -> tensor<128x128xf32>
  %C2 = linalg.generic {
      indexing_maps = [#map2, #bias, #map2],
      iterator_types = ["parallel", "parallel"]
    } ins(%C1, %bias : tensor<128x128xf32>, tensor<128xf32>)
      outs(%C1 : tensor<128x128xf32>) {
    ^bb0(%c: f32, %b: f32, %out: f32):
      %sum = arith.addf %c, %b : f32
      %relu = arith.maximumf %sum, %zero : f32
      linalg.yield %relu : f32
    } -> tensor<128x128xf32>
  return %C2 : tensor<128x128xf32>
}
