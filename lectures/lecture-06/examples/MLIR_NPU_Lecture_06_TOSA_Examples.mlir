// MLIR_NPU_Lecture_06_TOSA_Examples.mlir
// Educational pseudo-MLIR using generic-style TOSA operations for audit practice.

func.func @tosa_layer_norm_primitives(
    %x: tensor<1x128x768xf32>,
    %gamma: tensor<768xf32>,
    %beta: tensor<768xf32>) -> tensor<1x128x768xf32> {
  // mean = reduce_sum(x, axis=2) / 768
  %sum = "tosa.reduce_sum"(%x) {axis = 2 : i32}
    : (tensor<1x128x768xf32>) -> tensor<1x128x1xf32>
  %mean = "npu.pseudo_scale_by_inverse_k"(%sum) {k = 768 : i32}
    : (tensor<1x128x1xf32>) -> tensor<1x128x1xf32>

  // center = x - broadcast(mean)
  %mean_b = "tosa.tile"(%mean) {multiples = array<i64: 1, 1, 768>}
    : (tensor<1x128x1xf32>) -> tensor<1x128x768xf32>
  %center = "tosa.sub"(%x, %mean_b)
    : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>

  // variance = reduce_sum(center*center, axis=2) / 768
  %sq = "tosa.mul"(%center, %center) {shift = 0 : i8}
    : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
  %var_sum = "tosa.reduce_sum"(%sq) {axis = 2 : i32}
    : (tensor<1x128x768xf32>) -> tensor<1x128x1xf32>
  %var = "npu.pseudo_scale_by_inverse_k"(%var_sum) {k = 768 : i32}
    : (tensor<1x128x1xf32>) -> tensor<1x128x1xf32>
  %eps = "tosa.const"() {value = dense<0.00001> : tensor<1x128x1xf32>}
    : () -> tensor<1x128x1xf32>
  %var_eps = "tosa.add"(%var, %eps)
    : (tensor<1x128x1xf32>, tensor<1x128x1xf32>) -> tensor<1x128x1xf32>
  %inv = "tosa.rsqrt"(%var_eps)
    : (tensor<1x128x1xf32>) -> tensor<1x128x1xf32>
  %inv_b = "tosa.tile"(%inv) {multiples = array<i64: 1, 1, 768>}
    : (tensor<1x128x1xf32>) -> tensor<1x128x768xf32>

  // affine = center * inv * gamma + beta
  %norm = "tosa.mul"(%center, %inv_b) {shift = 0 : i8}
    : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
  %out = "npu.pseudo_affine_gamma_beta"(%norm, %gamma, %beta)
    : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
  return %out : tensor<1x128x768xf32>
}

func.func @tosa_gelu_approx_choices(%x: tensor<1x128x3072xf16>) -> tensor<1x128x3072xf16> {
  // Option A: tanh approximation if tosa.tanh is supported.
  // Option B: tosa.table for quantized activation LUT.
  // Option C: preserve graph and let NPU backend fuse to activation microcode.
  %out = "tosa.identity"(%x) : (tensor<1x128x3072xf16>) -> tensor<1x128x3072xf16>
  return %out : tensor<1x128x3072xf16>
}
