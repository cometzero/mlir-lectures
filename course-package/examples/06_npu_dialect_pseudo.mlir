// Pseudo NPU dialect. Design target, not upstream MLIR syntax.
module {
  npu.launch @matmul_i8 attributes {grid = [8, 16]} {
    npu.kernel {
      npu.dma_load %A_dram, %A_sram {bytes = 1024, double_buffer = true}
      npu.dma_load %B_dram, %B_sram {bytes = 1024, double_buffer = true}
      npu.barrier
      %acc = npu.matmul %A_sram, %B_sram, %acc_init
        {tile_m = 16, tile_n = 16, tile_k = 64, in_type = i8, acc_type = i32}
      %out = npu.requant %acc {multiplier = 12345, shift = 17, zp = 0}
      npu.dma_store %out, %C_dram {bytes = 256}
      npu.return
    }
  }
}
