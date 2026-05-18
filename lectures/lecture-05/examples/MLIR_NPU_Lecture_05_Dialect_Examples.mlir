// MLIR_NPU_Lecture_05_Dialect_Examples.mlir
// Educational examples for Lecture 05: Dialect System
// These examples intentionally mix real MLIR dialect names with pseudo NPU dialects.

//===----------------------------------------------------------------------===//
// 1. Common dialects in a high-level tensor program
//===----------------------------------------------------------------------===//

module {
  func.func @matmul_bias_relu(
      %arg0: tensor<1x128x256xf16>,
      %arg1: tensor<256x64xf16>,
      %bias: tensor<64xf16>) -> tensor<1x128x64xf16> {

    %init = tensor.empty() : tensor<1x128x64xf16>
    %mm = linalg.matmul
        ins(%arg0, %arg1 : tensor<1x128x256xf16>, tensor<256x64xf16>)
        outs(%init : tensor<1x128x64xf16>) -> tensor<1x128x64xf16>

    %zero = arith.constant 0.0 : f16
    %out = linalg.generic {
      indexing_maps = [
        affine_map<(b, m, n) -> (b, m, n)>,
        affine_map<(b, m, n) -> (n)>,
        affine_map<(b, m, n) -> (b, m, n)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%mm, %bias : tensor<1x128x64xf16>, tensor<64xf16>)
      outs(%init : tensor<1x128x64xf16>) {
    ^bb0(%v: f16, %b: f16, %dst: f16):
      %sum = arith.addf %v, %b : f16
      %relu = arith.maximumf %sum, %zero : f16
      linalg.yield %relu : f16
    } -> tensor<1x128x64xf16>

    return %out : tensor<1x128x64xf16>
  }
}

//===----------------------------------------------------------------------===//
// 2. Pseudo graph dialect after target-aware fusion/layout decisions
//===----------------------------------------------------------------------===//

"npu_graph.module"() ({
  "npu_graph.func"() ({
  ^bb0(%x: tensor<1x128x256xi8>, %w: tensor<256x64xi8>, %bias: tensor<64xi32>):
    %0 = "npu_graph.fusion_group"(%x, %w, %bias) {
      symbol_name = "matmul_bias_relu_0",
      fusion_kind = "matmul_bias_relu",
      input_layouts = ["NHK", "KN"],
      output_layout = "NHN",
      quant = "per_channel_i8_i32"
    } : (tensor<1x128x256xi8>, tensor<256x64xi8>, tensor<64xi32>) -> tensor<1x128x64xi8>
    "npu_graph.return"(%0) : (tensor<1x128x64xi8>) -> ()
  }) : () -> ()
}) : () -> ()

//===----------------------------------------------------------------------===//
// 3. Pseudo kernel dialect after tiling and memory planning
//===----------------------------------------------------------------------===//

"npu_kernel.module"() ({
  "npu_kernel.region"() ({
  ^tile(%tile_m: index, %tile_n: index, %tile_k: index):
    %sram_a = "npu_kernel.dma_load"(%tile_m, %tile_k) {
      memory_space = "sram_a",
      alignment = 64,
      burst = 128
    } : (index, index) -> !npu_kernel.sram_tile<16x32xi8>

    %sram_b = "npu_kernel.dma_load"(%tile_k, %tile_n) {
      memory_space = "sram_b",
      alignment = 64,
      burst = 128
    } : (index, index) -> !npu_kernel.sram_tile<32x16xi8>

    "npu_kernel.barrier"() {scope = "core"} : () -> ()

    %acc = "npu_kernel.matmul"(%sram_a, %sram_b) {
      tile_shape = [16, 16, 32],
      accumulator_type = "i32",
      array_shape = [16, 16]
    } : (!npu_kernel.sram_tile<16x32xi8>, !npu_kernel.sram_tile<32x16xi8>)
      -> !npu_kernel.acc_tile<16x16xi32>

    %out = "npu_kernel.requant_relu"(%acc) {
      scale_encoding = "per_channel",
      output_type = "i8"
    } : (!npu_kernel.acc_tile<16x16xi32>) -> !npu_kernel.sram_tile<16x16xi8>

    "npu_kernel.dma_store"(%out, %tile_m, %tile_n) {
      memory_space = "dram",
      alignment = 64
    } : (!npu_kernel.sram_tile<16x16xi8>, index, index) -> ()

    "npu_kernel.yield"() : () -> ()
  }) : () -> ()
}) : () -> ()

//===----------------------------------------------------------------------===//
// 4. Pseudo runtime dialect
//===----------------------------------------------------------------------===//

"npu_rt.module"() ({
  %cb = "npu_rt.create_command_buffer"() {
    abi_version = 1,
    max_commands = 4096
  } : () -> !npu_rt.command_buffer

  %desc_x = "npu_rt.tensor_desc"() {
    shape = [1, 128, 256],
    dtype = "i8",
    layout = "NHK",
    alignment = 64
  } : () -> !npu_rt.tensor_desc

  "npu_rt.launch"(%cb, %desc_x) {
    core_mask = 15,
    stream = 0,
    qos = "realtime"
  } : (!npu_rt.command_buffer, !npu_rt.tensor_desc) -> ()
}) : () -> ()
