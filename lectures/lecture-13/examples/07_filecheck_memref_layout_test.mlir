// RUN: mlir-opt %s --canonicalize | FileCheck %s
func.func @expect_sram_alloc() {
  // CHECK: memref.alloc{{.*}}memref<64x128xf16, 1>
  // CHECK-NOT: memref.copy
  %sram = memref.alloc() : memref<64x128xf16, 1>
  return
}
