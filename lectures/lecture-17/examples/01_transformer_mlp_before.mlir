// Lecture 17 example: Transformer MLP before fusion.
// This is pedagogical MLIR-like IR. Some custom attributes are pseudo.

#matmul_accesses = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (k, n)>,
  affine_map<(m, n, k) -> (m, n)>
]
#matmul_iterators = ["parallel", "parallel", "reduction"]

func.func @mlp_gelu(%x: tensor<128x4096xf16>,
                    %w1: tensor<4096x16384xf16>,
                    %b1: tensor<16384xf16>,
                    %w2: tensor<16384x4096xf16>,
                    %b2: tensor<4096xf16>) -> tensor<128x4096xf16> {
  %init1 = tensor.empty() : tensor<128x16384xf16>
  %h0 = linalg.generic {
      indexing_maps = #matmul_accesses,
      iterator_types = #matmul_iterators
    } ins(%x, %w1 : tensor<128x4096xf16>, tensor<4096x16384xf16>)
      outs(%init1 : tensor<128x16384xf16>) {
    ^bb0(%a: f16, %b: f16, %acc: f16):
      %p = arith.mulf %a, %b : f16
      %s = arith.addf %acc, %p : f16
      linalg.yield %s : f16
    } -> tensor<128x16384xf16>

  %h1 = linalg.generic {
      indexing_maps = [affine_map<(m, n) -> (m, n)>, affine_map<(m, n) -> (n)>, affine_map<(m, n) -> (m, n)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%h0, %b1 : tensor<128x16384xf16>, tensor<16384xf16>)
      outs(%init1 : tensor<128x16384xf16>) {
    ^bb0(%v: f16, %bias: f16, %out: f16):
      %r = arith.addf %v, %bias : f16
      linalg.yield %r : f16
    } -> tensor<128x16384xf16>

  %h2 = "npu_graph.gelu"(%h1) {approx = "tanh"} : (tensor<128x16384xf16>) -> tensor<128x16384xf16>

  %init2 = tensor.empty() : tensor<128x4096xf16>
  %y0 = linalg.generic {
      indexing_maps = #matmul_accesses,
      iterator_types = #matmul_iterators
    } ins(%h2, %w2 : tensor<128x16384xf16>, tensor<16384x4096xf16>)
      outs(%init2 : tensor<128x4096xf16>) {
    ^bb0(%a: f16, %b: f16, %acc: f16):
      %p = arith.mulf %a, %b : f16
      %s = arith.addf %acc, %p : f16
      linalg.yield %s : f16
    } -> tensor<128x4096xf16>

  %y = linalg.generic {
      indexing_maps = [affine_map<(m, n) -> (m, n)>, affine_map<(m, n) -> (n)>, affine_map<(m, n) -> (m, n)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%y0, %b2 : tensor<128x4096xf16>, tensor<4096xf16>)
      outs(%init2 : tensor<128x4096xf16>) {
    ^bb0(%v: f16, %bias: f16, %out: f16):
      %r = arith.addf %v, %bias : f16
      linalg.yield %r : f16
    } -> tensor<128x4096xf16>

  return %y : tensor<128x4096xf16>
}
