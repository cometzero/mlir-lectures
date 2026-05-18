// RUN: mlir-opt %s --pass-pipeline="builtin.module(npu-capstone-pipeline)" | FileCheck %s

func.func @matmul_bias_relu(%a: tensor<64x32xi8>, %b: tensor<32x64xi8>,
                            %bias: tensor<64xi32>) -> tensor<64x64xi8> {
  // Educational placeholder input.
  // CHECK-LABEL: func.func @matmul_bias_relu
  // CHECK: npu.matmul
  // CHECK-SAME: m = 16
  // CHECK-SAME: n = 16
  // CHECK-SAME: k = 32
  // CHECK: npu.epilogue
  // CHECK-SAME: kind = "bias_relu"
  // CHECK: npu.dma.store
  %0 = "test.placeholder"(%a, %b, %bias) : (tensor<64x32xi8>, tensor<32x64xi8>, tensor<64xi32>) -> tensor<64x64xi8>
  return %0 : tensor<64x64xi8>
}

// -----

// RUN: not mlir-opt %s --npu-verify-kernel 2>&1 | FileCheck %s --check-prefix=ERR
module {
  // ERR: error: npu.matmul tile does not fit target native tile or tail policy
  "npu.matmul"() {m = 17 : i64, n = 16 : i64, k = 32 : i64} : () -> ()
}
