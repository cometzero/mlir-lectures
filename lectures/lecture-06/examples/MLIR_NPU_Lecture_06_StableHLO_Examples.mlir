// MLIR_NPU_Lecture_06_StableHLO_Examples.mlir
// Educational pseudo-MLIR. Adapt syntax to your exact StableHLO build.

// Example 1: MatMul + Bias + GELU exact decomposition.
// Goal: recognize high-level semantics before NPU legalization.
func.func @matmul_bias_gelu_exact(
    %x: tensor<1x128x768xf16>,
    %w: tensor<768x3072xf16>,
    %bias: tensor<3072xf16>) -> tensor<1x128x3072xf16> {
  %mm = stablehlo.dot_general %x, %w,
    contracting_dims = [2] x [0]
    : (tensor<1x128x768xf16>, tensor<768x3072xf16>) -> tensor<1x128x3072xf16>

  %b = stablehlo.broadcast_in_dim %bias, dims = [2]
    : (tensor<3072xf16>) -> tensor<1x128x3072xf16>
  %z = stablehlo.add %mm, %b : tensor<1x128x3072xf16>

  // GELU exact: 0.5 * z * (1 + erf(z / sqrt(2))).
  %c_half = stablehlo.constant dense<0.5> : tensor<f16>
  %c_one  = stablehlo.constant dense<1.0> : tensor<f16>
  %c_is2  = stablehlo.constant dense<0.70710678118> : tensor<f16>
  %half = stablehlo.broadcast_in_dim %c_half, dims = [] : (tensor<f16>) -> tensor<1x128x3072xf16>
  %one  = stablehlo.broadcast_in_dim %c_one,  dims = [] : (tensor<f16>) -> tensor<1x128x3072xf16>
  %is2  = stablehlo.broadcast_in_dim %c_is2,  dims = [] : (tensor<f16>) -> tensor<1x128x3072xf16>
  %t0 = stablehlo.multiply %z, %is2 : tensor<1x128x3072xf16>
  %t1 = stablehlo.erf %t0 : tensor<1x128x3072xf16>
  %t2 = stablehlo.add %one, %t1 : tensor<1x128x3072xf16>
  %t3 = stablehlo.multiply %z, %t2 : tensor<1x128x3072xf16>
  %out = stablehlo.multiply %half, %t3 : tensor<1x128x3072xf16>
  return %out : tensor<1x128x3072xf16>
}

// Example 2: LayerNorm decomposition over last dimension K=768.
// Actual stablehlo.reduce syntax has a region; this file keeps it compact for reading.
func.func @layer_norm_decomposed(
    %x: tensor<1x128x768xf32>,
    %gamma: tensor<768xf32>,
    %beta: tensor<768xf32>) -> tensor<1x128x768xf32> {
  %sum = stablehlo.reduce(%x) applies stablehlo.add across dimensions = [2]
    : tensor<1x128x768xf32> -> tensor<1x128xf32>
  %mean = stablehlo.divide %sum, dense<768.0> : tensor<1x128xf32>
  %mean_b = stablehlo.broadcast_in_dim %mean, dims = [0, 1]
    : (tensor<1x128xf32>) -> tensor<1x128x768xf32>
  %center = stablehlo.subtract %x, %mean_b : tensor<1x128x768xf32>
  %sq = stablehlo.multiply %center, %center : tensor<1x128x768xf32>
  %var_sum = stablehlo.reduce(%sq) applies stablehlo.add across dimensions = [2]
    : tensor<1x128x768xf32> -> tensor<1x128xf32>
  %var = stablehlo.divide %var_sum, dense<768.0> : tensor<1x128xf32>
  %var_eps = stablehlo.add %var, dense<0.00001> : tensor<1x128xf32>
  %inv = stablehlo.rsqrt %var_eps : tensor<1x128xf32>
  %inv_b = stablehlo.broadcast_in_dim %inv, dims = [0, 1]
    : (tensor<1x128xf32>) -> tensor<1x128x768xf32>
  %norm = stablehlo.multiply %center, %inv_b : tensor<1x128x768xf32>
  %g = stablehlo.broadcast_in_dim %gamma, dims = [2]
    : (tensor<768xf32>) -> tensor<1x128x768xf32>
  %bt = stablehlo.broadcast_in_dim %beta, dims = [2]
    : (tensor<768xf32>) -> tensor<1x128x768xf32>
  %scaled = stablehlo.multiply %norm, %g : tensor<1x128x768xf32>
  %out = stablehlo.add %scaled, %bt : tensor<1x128x768xf32>
  return %out : tensor<1x128x768xf32>
}
