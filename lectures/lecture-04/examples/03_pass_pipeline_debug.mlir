// Lecture 04 - Example 03: pass pipeline debug commands.
// Try these after building LLVM/MLIR:
//   mlir-opt %s --pass-pipeline='builtin.module(func.func(cse,canonicalize))'
//   mlir-opt %s --pass-pipeline='builtin.module(func.func(cse,canonicalize))' --mlir-print-ir-after-all
//   mlir-opt %s --pass-pipeline='builtin.module(func.func(cse,canonicalize))' --mlir-print-ir-after-change
//   mlir-opt %s --mlir-disable-threading --pass-pipeline='builtin.module(func.func(cse,canonicalize))' --mlir-timing

func.func @debug_me() -> (i32, i32) {
  %0 = arith.constant 1 : i32
  %1 = arith.constant 1 : i32
  return %0, %1 : i32, i32
}
