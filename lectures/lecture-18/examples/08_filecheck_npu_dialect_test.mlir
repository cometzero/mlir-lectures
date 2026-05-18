// RUN: npu-opt %s | FileCheck %s
// RUN: npu-opt %s --mlir-print-op-generic | FileCheck %s --check-prefix=GENERIC

// CHECK-LABEL: func.func @round_trip_matmul
// CHECK: npu.matmul
// CHECK-SAME: tile = #npu.tile<m = 16, n = 16, k = 32>
// GENERIC: "npu.matmul"
func.func @round_trip_matmul(%lhs: !npu.tile_buffer<i8, 16x32, #npu.layout<mk>>, %rhs: !npu.tile_buffer<i8, 32x16, #npu.layout<kn>>, %acc: !npu.acc_buffer<i32, 16x16>) {
  %out = npu.matmul %lhs, %rhs, %acc {tile = #npu.tile<m = 16, n = 16, k = 32>, lhs_layout = #npu.layout<mk>, rhs_layout = #npu.layout<kn>} : !npu.tile_buffer<i8, 16x32, #npu.layout<mk>>, !npu.tile_buffer<i8, 32x16, #npu.layout<kn>>, !npu.acc_buffer<i32, 16x16> -> !npu.acc_buffer<i32, 16x16>
  return
}
