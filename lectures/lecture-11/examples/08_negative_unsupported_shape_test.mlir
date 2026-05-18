// RUN: mlir-opt %s --npu-legalize-graph -verify-diagnostics

module {
  func.func @bad_k_tile(%a: tensor<128x250xf16>,
                        %b: tensor<250x128xf16>,
                        %bias: tensor<128xf16>) -> tensor<128x128xf16> {
    %init = tensor.empty() : tensor<128x128xf16>
    // expected-error @+1 {{cannot legalize matmul: K dimension 250 is not compatible with NPU MMA tile K=32}}
    %mm = linalg.matmul ins(%a, %b : tensor<128x250xf16>, tensor<250x128xf16>)
                       outs(%init : tensor<128x128xf16>) -> tensor<128x128xf16>
    return %mm : tensor<128x128xf16>
  }
}
