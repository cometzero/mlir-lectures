// Educational pseudo IR for Lecture 20 Capstone.
// Not intended to compile in upstream MLIR without defining the dialect.
module {
  npu_rt.module @mlp_capstone {
    npu_rt.entry @main(%x: !npu_rt.tensor_desc, %w1: !npu_rt.tensor_desc,
                       %b1: !npu_rt.tensor_desc, %w2: !npu_rt.tensor_desc,
                       %out: !npu_rt.tensor_desc) {
      %cmd = npu_rt.command_buffer {version = 1 : i32} {
        %a = npu.dma.load %x  {space = "dram_to_sram", bytes = 65536 : i64}
        %b = npu.dma.load %w1 {space = "dram_to_sram", bytes = 131072 : i64}
        %t0 = npu.barrier.await %a, %b
        %c = npu.matmul %a, %b {m = 16 : i64, n = 16 : i64, k = 32 : i64,
                                lhs_layout = "mk", rhs_layout = "kn",
                                acc_type = "i32"} after %t0
        %e = npu.epilogue %c, %b1 {kind = "bias_gelu", approx = "tanh"}
        %d = npu.dma.store %e, %out {space = "sram_to_dram"}
        npu_rt.yield %d
      }
      npu_rt.launch %cmd : !npu_rt.command_buffer
      npu_rt.return
    }
  }
}
