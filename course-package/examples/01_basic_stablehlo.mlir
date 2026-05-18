// Basic IR reading example: operation, value, type, attribute.
module {
  func.func @add_relu(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
    %0 = stablehlo.add %a, %b : tensor<4xf32>
    %z = stablehlo.constant dense<0.0> : tensor<4xf32>
    %1 = stablehlo.maximum %0, %z : tensor<4xf32>
    return %1 : tensor<4xf32>
  }
}
