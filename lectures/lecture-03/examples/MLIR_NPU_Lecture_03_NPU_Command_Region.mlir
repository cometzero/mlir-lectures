// MLIR_NPU_Lecture_03_NPU_Command_Region.mlir
// Pseudo IR for Lecture 03. This is not an upstream MLIR dialect.
// Goal: practice region/block design for an embedded/edge NPU compiler.

// -----------------------------------------------------------------------------
// Pseudo custom op: npu.kernel owns a kernel body region.
// Region arguments model descriptors and compile/runtime launch parameters.
// -----------------------------------------------------------------------------
"npu.kernel"() ({
^entry(%A: !npu.tensor_desc, %B: !npu.tensor_desc, %C: !npu.tensor_desc,
       %M: index, %N: index, %K: index):
  %tile = "npu.tile_config"() {
    m = 128 : i64,
    n = 128 : i64,
    k = 64 : i64,
    accumulator = "i32",
    dataflow = "output_stationary"
  } : () -> !npu.tile_config

  // A structured loop can be lowered later into command-buffer blocks.
  "npu.for_tiles"(%M, %N, %K) ({
  ^tile_body(%m0: index, %n0: index, %k0: index):
    %a_sram, %ta = "npu.dma.async"(%A, %m0, %k0) {
      src_space = "dram", dst_space = "sram", bytes = 32768 : i64
    } : (!npu.tensor_desc, index, index) -> (!npu.sram_view, !npu.token)

    %b_sram, %tb = "npu.dma.async"(%B, %k0, %n0) {
      src_space = "dram", dst_space = "sram", bytes = 32768 : i64
    } : (!npu.tensor_desc, index, index) -> (!npu.sram_view, !npu.token)

    "npu.await"(%ta, %tb) : (!npu.token, !npu.token) -> ()
    %acc = "npu.matmul"(%a_sram, %b_sram, %tile) {
      layout_a = "blocked_mk", layout_b = "blocked_kn"
    } : (!npu.sram_view, !npu.sram_view, !npu.tile_config) -> !npu.acc_view

    %out_sram = "npu.requant_relu"(%acc) {
      scale = 0.0078125 : f32,
      zero_point = 0 : i32,
      dst_type = "i8"
    } : (!npu.acc_view) -> !npu.sram_view

    %ts = "npu.dma.async"(%out_sram, %C, %m0, %n0) {
      src_space = "sram", dst_space = "dram", bytes = 16384 : i64
    } : (!npu.sram_view, !npu.tensor_desc, index, index) -> !npu.token

    "npu.await"(%ts) : (!npu.token) -> ()
    "npu.yield"() : () -> ()
  }) : (index, index, index) -> ()

  "npu.return"() : () -> ()
}) : () -> ()

// -----------------------------------------------------------------------------
// Verification questions for the pseudo IR:
// 1. Does every block have a terminator? npu.yield / npu.return.
// 2. Are region arguments typed and scoped correctly?
// 3. Are async tokens consumed before using SRAM views?
// 4. Are command attributes compile-time metadata rather than runtime operands?
// 5. Can this region be serialized into a deterministic command buffer?
