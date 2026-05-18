// RUN: mlir-opt %s --transform-interpreter --canonicalize | FileCheck %s
// NOTE: This is a template. Adjust exact CHECK lines after choosing an MLIR version.

// CHECK-LABEL: func.func @matmul_bias_relu
// CHECK: scf.for
// CHECK: vector.contract
// CHECK-NOT: memref.copy

module attributes {transform.with_named_sequence} {
  // Payload and transform schedule would be placed here in a real test.
}
