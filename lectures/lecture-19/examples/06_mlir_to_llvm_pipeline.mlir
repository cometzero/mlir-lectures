// RUN: mlir-opt %s --pass-pipeline='builtin.module(convert-npu-rt-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | FileCheck %s

module {
  func.func @main(%arg0: memref<1x4096xi8>, %arg1: memref<1x4096xi8>) {
    // Pseudo op before conversion:
    // npu_rt.submit @matmul_tile(%arg0, %arg1) : memref<1x4096xi8>, memref<1x4096xi8>
    return
  }
}

// CHECK-LABEL: llvm.func @main
// CHECK: llvm.call @npuSubmit
// CHECK: llvm.return
