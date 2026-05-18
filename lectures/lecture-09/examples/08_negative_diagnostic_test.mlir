// RUN: mlir-opt %s --pass-pipeline='builtin.module(func.func(npu-fuse-matmul-epilogue))' -verify-diagnostics

module {
  func.func @bad_dynamic(%A: tensor<?x64xf32>, %B: tensor<64x128xf32>) -> tensor<?x128xf32> {
    // expected-error @below {{illegal dynamic M dimension for target NPU tile policy}}
    %0 = "test.matmul"(%A, %B) : (tensor<?x64xf32>, tensor<64x128xf32>) -> tensor<?x128xf32>
    return %0 : tensor<?x128xf32>
  }
}
