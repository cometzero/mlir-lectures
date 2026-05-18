// Lecture 04 - Example 02: CSE regression test.
// RUN: mlir-opt %s --pass-pipeline='builtin.module(func.func(cse))' | FileCheck %s

// CHECK-LABEL: func.func @simple_constant
func.func @simple_constant() -> (i32, i32) {
  // CHECK-NEXT: %[[C:.*]] = arith.constant 1
  // CHECK-NEXT: return %[[C]], %[[C]]
  %0 = arith.constant 1 : i32
  %1 = arith.constant 1 : i32
  return %0, %1 : i32, i32
}
