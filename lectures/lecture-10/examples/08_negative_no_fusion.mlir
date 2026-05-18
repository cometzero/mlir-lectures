// RUN: npu-opt %s --npu-fuse-matmul-epilogue | FileCheck %s

// Bias is consumed by two users. A conservative fusion pattern should reject this.
// CHECK-LABEL: func.func @no_fusion_multiple_users
// CHECK: linalg.matmul
// CHECK-NOT: "npu_graph.matmul"

func.func @no_fusion_multiple_users(%A: tensor<128x256xf16>,
                                    %B: tensor<256x128xf16>,
                                    %bias: tensor<128xf16>) -> (tensor<128x128xf16>, tensor<128x128xf16>) {
  %c0 = arith.constant 0.0 : f16
  %init = tensor.empty() : tensor<128x128xf16>
  %zero = linalg.fill ins(%c0 : f16) outs(%init : tensor<128x128xf16>) -> tensor<128x128xf16>
  %mm = linalg.matmul ins(%A, %B : tensor<128x256xf16>, tensor<256x128xf16>)
                      outs(%zero : tensor<128x128xf16>) -> tensor<128x128xf16>
  // Placeholder for two consumers of %mm. Keep unfused until a safe duplication policy exists.
  return %mm, %mm : tensor<128x128xf16>, tensor<128x128xf16>
}
