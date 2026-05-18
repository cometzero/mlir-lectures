// RUN: npu-opt %s | FileCheck %s
// Educational pseudo test.

// CHECK-LABEL: func.func @tile_command_sequence
// CHECK: npu.dma
// CHECK: npu.barrier
// CHECK: npu.matmul
func.func @tile_command_sequence(%lhs: !npu.tile_buffer<i8, 16x32, #npu.layout<mk>>,
                                 %rhs: !npu.tile_buffer<i8, 32x16, #npu.layout<kn>>,
                                 %acc: !npu.acc_buffer<i32, 16x16>,
                                 %tok0: !npu.token, %tok1: !npu.token) {
  npu.barrier %tok0, %tok1 {scope = #npu.barrier<dma>} : !npu.token, !npu.token
  %out = npu.matmul %lhs, %rhs, %acc
    {tile = #npu.tile<m = 16, n = 16, k = 32>, lhs_layout = #npu.layout<mk>, rhs_layout = #npu.layout<kn>}
    : !npu.tile_buffer<i8, 16x32, #npu.layout<mk>>, !npu.tile_buffer<i8, 32x16, #npu.layout<kn>>, !npu.acc_buffer<i32, 16x16> -> !npu.acc_buffer<i32, 16x16>
  return
}
