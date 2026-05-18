// Lecture 04 - Example 06: mlir-translate from MLIR LLVM dialect to LLVM IR.
// Requires a build with MLIR LLVM translation registered.
// RUN: mlir-translate --mlir-to-llvmir %s | FileCheck %s

llvm.func @add_i32(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK-LABEL: define i32 @add_i32
  // CHECK: add i32
  %0 = llvm.add %arg0, %arg1 : i32
  llvm.return %0 : i32
}
