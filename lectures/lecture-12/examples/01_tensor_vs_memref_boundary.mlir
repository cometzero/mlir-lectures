// Lecture 12 example: tensor/memref boundary.
// This file is intentionally educational and may require a locally built MLIR.

func.func @tensor_world(%A: tensor<64x64xf32>, %B: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %C0 = tensor.empty() : tensor<64x64xf32>
  %C = linalg.matmul ins(%A, %B : tensor<64x64xf32>, tensor<64x64xf32>)
                    outs(%C0 : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %C : tensor<64x64xf32>
}

func.func @memref_world(%A: memref<64x64xf32>, %B: memref<64x64xf32>, %C: memref<64x64xf32>) {
  linalg.matmul ins(%A, %B : memref<64x64xf32>, memref<64x64xf32>)
                outs(%C : memref<64x64xf32>)
  return
}
