// Educational pseudo dialect. Not upstream MLIR.
module {
  npu_rt.module @mlp_tile {
    abi_version = "1.0",
    target = "npu256_edge_v1",
    features = ["i8_gemm", "requant", "dma_overlap"]
  } {
    npu_rt.tensor_desc @A {
      index = 0 : i32,
      shape = [1, 4096],
      stride = [4096, 1],
      dtype = "i8",
      layout = "row_major",
      memory_space = "dram",
      quant = {scale = 0.03125 : f32, zero_point = 0 : i32}
    }

    npu_rt.tensor_desc @B {
      index = 1 : i32,
      shape = [4096, 4096],
      stride = [4096, 1],
      dtype = "i4_packed",
      layout = "k_major_n_blocked",
      memory_space = "const"
    }

    npu_rt.tensor_desc @C {
      index = 2 : i32,
      shape = [1, 4096],
      stride = [4096, 1],
      dtype = "i8",
      layout = "row_major",
      memory_space = "dram",
      quant = {scale = 0.0625 : f32, zero_point = 0 : i32}
    }

    npu_rt.command_buffer @matmul_tile {
      %t0 = npu_rt.dma_load @A -> @sram_A {bytes = 512 : i64}
      %t1 = npu_rt.dma_load @B -> @sram_B {bytes = 4096 : i64}
      npu_rt.wait %t0, %t1
      %t2 = npu_rt.matmul @sram_A, @sram_B -> @acc {
        tile_m = 16 : i32, tile_n = 16 : i32, tile_k = 32 : i32,
        lhs_dtype = "i8", rhs_dtype = "i4", acc_dtype = "i32"
      }
      %t3 = npu_rt.requant %t2 -> @sram_C {multiplier = 1073741824 : i32, shift = 31 : i32}
      %t4 = npu_rt.dma_store @sram_C -> @C {bytes = 16 : i64}
      npu_rt.fence %t4 {name = "done"}
    }
  }
}
