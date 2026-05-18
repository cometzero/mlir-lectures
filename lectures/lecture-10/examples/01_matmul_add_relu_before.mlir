// Lecture 10 example: MatMul + Bias + ReLU before fusion.
// This is intended for reading and FileCheck-style testing.

#matmul_accesses = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (k, n)>,
  affine_map<(m, n, k) -> (m, n)>
]
#bias_accesses = [
  affine_map<(m, n) -> (m, n)>,
  affine_map<(m, n) -> (n)>,
  affine_map<(m, n) -> (m, n)>
]
#eltwise_accesses = [
  affine_map<(m, n) -> (m, n)>,
  affine_map<(m, n) -> (m, n)>
]

func.func @matmul_bias_relu(%A: tensor<128x256xf16>,
                            %B: tensor<256x128xf16>,
                            %bias: tensor<128xf16>) -> tensor<128x128xf16> {
  %c0 = arith.constant 0.0 : f16
  %init = tensor.empty() : tensor<128x128xf16>
  %zero = linalg.fill ins(%c0 : f16)
                      outs(%init : tensor<128x128xf16>) -> tensor<128x128xf16>

  %mm = linalg.matmul ins(%A, %B : tensor<128x256xf16>, tensor<256x128xf16>)
                      outs(%zero : tensor<128x128xf16>) -> tensor<128x128xf16>

  %add = linalg.generic {
      indexing_maps = #bias_accesses,
      iterator_types = ["parallel", "parallel"]}
      ins(%mm, %bias : tensor<128x128xf16>, tensor<128xf16>)
      outs(%init : tensor<128x128xf16>) {
    ^bb0(%x: f16, %b: f16, %out: f16):
      %sum = arith.addf %x, %b : f16
      linalg.yield %sum : f16
  } -> tensor<128x128xf16>

  %relu = linalg.generic {
      indexing_maps = #eltwise_accesses,
      iterator_types = ["parallel", "parallel"]}
      ins(%add : tensor<128x128xf16>)
      outs(%init : tensor<128x128xf16>) {
    ^bb0(%x: f16, %out: f16):
      %z = arith.constant 0.0 : f16
      %r = arith.maximumf %x, %z : f16
      linalg.yield %r : f16
  } -> tensor<128x128xf16>

  return %relu : tensor<128x128xf16>
}
