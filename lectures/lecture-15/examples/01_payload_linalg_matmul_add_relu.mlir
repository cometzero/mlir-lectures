// Payload IR for Lecture 15.
// Goal: schedule MatMul + Bias + ReLU using Transform Dialect.
// This file is intentionally small so that FileCheck-based tests are easy.

#map_matmul_a = affine_map<(m, n, k) -> (m, k)>
#map_matmul_b = affine_map<(m, n, k) -> (k, n)>
#map_matmul_c = affine_map<(m, n, k) -> (m, n)>

module attributes {transform.with_named_sequence} {
  func.func @matmul_bias_relu(
      %A: tensor<128x64xi8>,
      %B: tensor<64x128xi8>,
      %Bias: tensor<128xi32>) -> tensor<128x128xi32> {
    %empty = tensor.empty() : tensor<128x128xi32>
    %zero = arith.constant 0 : i32
    %init = linalg.fill ins(%zero : i32) outs(%empty : tensor<128x128xi32>) -> tensor<128x128xi32>
    %mm = linalg.matmul ins(%A, %B : tensor<128x64xi8>, tensor<64x128xi8>)
                       outs(%init : tensor<128x128xi32>) -> tensor<128x128xi32>
    // Bias + ReLU is shown as pseudo linalg.generic for teaching.
    %out = "npu_graph.bias_relu"(%mm, %Bias) : (tensor<128x128xi32>, tensor<128xi32>) -> tensor<128x128xi32>
    return %out : tensor<128x128xi32>
  }
}
