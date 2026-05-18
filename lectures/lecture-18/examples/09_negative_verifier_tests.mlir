// RUN: npu-opt %s -verify-diagnostics

func.func @bad_tile(%lhs: !npu.tile_buffer<i8, 8x32, #npu.layout<mk>>, %rhs: !npu.tile_buffer<i8, 32x16, #npu.layout<kn>>, %acc: !npu.acc_buffer<i32, 8x16>) {
  // expected-error @+1 {{unsupported native tile shape}}
  %out = npu.matmul %lhs, %rhs, %acc {tile = #npu.tile<m = 8, n = 16, k = 32>, lhs_layout = #npu.layout<mk>, rhs_layout = #npu.layout<kn>} : !npu.tile_buffer<i8, 8x32, #npu.layout<mk>>, !npu.tile_buffer<i8, 32x16, #npu.layout<kn>>, !npu.acc_buffer<i32, 8x16> -> !npu.acc_buffer<i32, 8x16>
  return
}

func.func @empty_barrier() {
  // expected-error @+1 {{requires at least one token operand}}
  npu.barrier {scope = #npu.barrier<dma>} :
  return
}
