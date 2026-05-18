// RUN: mlir-opt %s --npu-legalize-graph | FileCheck %s

// CHECK-LABEL: func.func @matmul_bias_gelu
// CHECK-NOT: npu_graph.gelu_exact
// CHECK: "npu_graph.matmul_epilogue"
// CHECK-SAME: epilogue = "gelu_lut"
// CHECK-SAME: k_tile = 32

module {
  func.func @matmul_bias_gelu(%a: tensor<128x256xf16>,
                              %b: tensor<256x128xf16>,
                              %bias: tensor<128xf16>) -> tensor<128x128xf16> {
    %init = tensor.empty() : tensor<128x128xf16>
    %mm = linalg.matmul ins(%a, %b : tensor<128x256xf16>, tensor<256x128xf16>)
                       outs(%init : tensor<128x128xf16>) -> tensor<128x128xf16>
    %biased = "npu_graph.broadcast_add"(%mm, %bias)
      : (tensor<128x128xf16>, tensor<128xf16>) -> tensor<128x128xf16>
    %gelu = "npu_graph.gelu_exact"(%biased)
      {mode = "erf"} : (tensor<128x128xf16>) -> tensor<128x128xf16>
    return %gelu : tensor<128x128xf16>
  }
}
