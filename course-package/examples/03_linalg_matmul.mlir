// Linalg-on-tensors matmul.
module {
  func.func @matmul(%A: tensor<128x64xf32>, %B: tensor<64x256xf32>, %C: tensor<128x256xf32>) -> tensor<128x256xf32> {
    %0 = linalg.matmul ins(%A, %B : tensor<128x64xf32>, tensor<64x256xf32>)
                       outs(%C : tensor<128x256xf32>) -> tensor<128x256xf32>
    return %0 : tensor<128x256xf32>
  }
}
