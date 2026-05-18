// RUN: mlir-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: func.func @scf_for_basic
// CHECK: scf.for
// CHECK: memref.load
// CHECK: memref.store

func.func @scf_for_basic(%A: memref<128xf32>, %B: memref<128xf32>) {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c128 step %c1 {
    %a = memref.load %A[%i] : memref<128xf32>
    memref.store %a, %B[%i] : memref<128xf32>
  }
  return
}
