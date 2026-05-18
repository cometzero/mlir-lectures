// alloc_tensor creates a fresh tensor SSA chain that can resolve alias conflicts.

func.func @fresh_buffer_anchor(%A: tensor<128x128xf32>, %B: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %dst = bufferization.alloc_tensor() : tensor<128x128xf32>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%A, %B : tensor<128x128xf32>, tensor<128x128xf32>)
      outs(%dst : tensor<128x128xf32>) {
    ^bb0(%a: f32, %b: f32, %out: f32):
      %s = arith.addf %a, %b : f32
      linalg.yield %s : f32
    } -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}
