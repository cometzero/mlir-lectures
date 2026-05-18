// Lecture 04 - Example 01: parse and verify only.
// RUN: mlir-opt %s | FileCheck %s

// CHECK-LABEL: func.func @identity
func.func @identity(%x: i32) -> i32 {
  func.return %x : i32
}
