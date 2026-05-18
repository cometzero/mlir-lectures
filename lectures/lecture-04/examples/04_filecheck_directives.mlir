// Lecture 04 - Example 04: FileCheck directive patterns.
// RUN: mlir-opt %s --pass-pipeline='builtin.module(func.func(cse))' | FileCheck %s

// CHECK-LABEL: func.func @two_results
// CHECK-SAME:  () -> (i32, i32)
func.func @two_results() -> (i32, i32) {
  // CHECK-DAG: %[[A:.*]] = arith.constant 1
  // CHECK-DAG: %[[B:.*]] = arith.constant 2
  // CHECK-NOT: arith.constant 3
  // CHECK: return %[[A]], %[[B]]
  %0 = arith.constant 1 : i32
  %1 = arith.constant 2 : i32
  return %0, %1 : i32, i32
}
