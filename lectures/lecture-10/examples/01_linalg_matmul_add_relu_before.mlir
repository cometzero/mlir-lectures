// Lecture 10 - illustrative input IR for pattern matching.
// This file is intentionally compact; exact op names may vary by MLIR version.

#matmulA = affine_map<(m, n, k) -> (m, k)>
#matmulB = affine_map<(m, n, k) -> (k, n)>
#matmulC = affine_map<(m, n, k) -> (m, n)>
#eltwise2 = affine_map<(m, n) -> (m, n)>
#biasMap = affine_map<(m, n) -> (n)>

func.func @matmul_bias_relu(
    %A : tensor<128x256xf32>,
    %B : tensor<256x512xf32>,
    %bias : tensor<512xf32>) -> tensor<128x512xf32> {
  %c0 = arith.constant 0.0 : f32
  %empty0 = tensor.empty() : tensor<128x512xf32>
  %init = linalg.fill ins(%c0 : f32) outs(%empty0 : tensor<128x512xf32>) -> tensor<128x512xf32>

  %mm = linalg.generic {
      indexing_maps = [#matmulA, #matmulB, #matmulC],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%A, %B : tensor<128x256xf32>, tensor<256x512xf32>)
      outs(%init : tensor<128x512xf32>) {
    ^bb0(%a: f32, %b: f32, %acc: f32):
      %mul = arith.mulf %a, %b : f32
      %add = arith.addf %acc, %mul : f32
      linalg.yield %add : f32
  } -> tensor<128x512xf32>

  %empty1 = tensor.empty() : tensor<128x512xf32>
  %add_bias = linalg.generic {
      indexing_maps = [#eltwise2, #biasMap, #eltwise2],
      iterator_types = ["parallel", "parallel"]}
      ins(%mm, %bias : tensor<128x512xf32>, tensor<512xf32>)
      outs(%empty1 : tensor<128x512xf32>) {
    ^bb0(%x: f32, %b: f32, %out: f32):
      %y = arith.addf %x, %b : f32
      linalg.yield %y : f32
  } -> tensor<128x512xf32>

  %empty2 = tensor.empty() : tensor<128x512xf32>
  %relu = linalg.generic {
      indexing_maps = [#eltwise2, #eltwise2],
      iterator_types = ["parallel", "parallel"]}
      ins(%add_bias : tensor<128x512xf32>)
      outs(%empty2 : tensor<128x512xf32>) {
    ^bb0(%x: f32, %out: f32):
      %y = arith.maxnumf %x, %c0 : f32
      linalg.yield %y : f32
  } -> tensor<128x512xf32>

  return %relu : tensor<128x512xf32>
}
