// Lecture 18 - Custom NPU Dialect overview
// Educational pseudo IR. Requires a custom NPU dialect implementation.

module {
  func.func @tile_command(%A: memref<128x128xi8, #npu.mem_space<dram>>,
                          %B: memref<128x128xi8, #npu.mem_space<dram>>,
                          %C: memref<128x128xi32, #npu.mem_space<dram>>) {
    %sa = npu.alloc_sram {bytes = 512 : i64} : !npu.tile_buffer<i8, 16x32, #npu.layout<mk>>
    %sb = npu.alloc_sram {bytes = 512 : i64} : !npu.tile_buffer<i8, 32x16, #npu.layout<kn>>
    %acc = npu.alloc_accum : !npu.acc_buffer<i32, 16x16>

    %ta = npu.dma %A[0, 0] -> %sa {direction = #npu.dma<dram_to_sram>, elements = 512 : i64}
      : memref<128x128xi8, #npu.mem_space<dram>> to !npu.tile_buffer<i8, 16x32, #npu.layout<mk>>
    %tb = npu.dma %B[0, 0] -> %sb {direction = #npu.dma<dram_to_sram>, elements = 512 : i64}
      : memref<128x128xi8, #npu.mem_space<dram>> to !npu.tile_buffer<i8, 32x16, #npu.layout<kn>>
    npu.barrier %ta, %tb {scope = #npu.barrier<dma>} : !npu.token, !npu.token

    %out = npu.matmul %sa, %sb, %acc
      {tile = #npu.tile<m = 16, n = 16, k = 32>, lhs_layout = #npu.layout<mk>, rhs_layout = #npu.layout<kn>}
      : !npu.tile_buffer<i8, 16x32, #npu.layout<mk>>, !npu.tile_buffer<i8, 32x16, #npu.layout<kn>>, !npu.acc_buffer<i32, 16x16> -> !npu.acc_buffer<i32, 16x16>

    %tc = npu.dma %out -> %C[0, 0] {direction = #npu.dma<accum_to_dram>, elements = 256 : i64}
      : !npu.acc_buffer<i32, 16x16> to memref<128x128xi32, #npu.mem_space<dram>>
    npu.barrier %tc {scope = #npu.barrier<dma>} : !npu.token
    return
  }
}
