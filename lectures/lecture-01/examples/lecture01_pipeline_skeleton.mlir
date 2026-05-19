// Lecture 01 pseudo MLIR skeleton. Use for reading the pipeline, not for final syntax.
// Try from the repository root:
// mlir-opt --allow-unregistered-dialect lectures/lecture-01/examples/lecture01_pipeline_skeleton.mlir

module {
  func.func @matmul_bias_relu(%A: tensor<1x128x256xf32>, %B: tensor<256x512xf32>,
                              %bias: tensor<512xf32>, %zero: tensor<1x128x512xf32>)
      -> tensor<1x128x512xf32> {
    %0 = "stablehlo.dot_general"(%A, %B) {dimension_numbers = "batch=[0], lhs_contracting=[2], rhs_contracting=[0]"}
      : (tensor<1x128x256xf32>, tensor<256x512xf32>) -> tensor<1x128x512xf32>
    %1 = "stablehlo.add"(%0, %bias) : (tensor<1x128x512xf32>, tensor<512xf32>) -> tensor<1x128x512xf32>
    %2 = "stablehlo.maximum"(%1, %zero) : (tensor<1x128x512xf32>, tensor<1x128x512xf32>) -> tensor<1x128x512xf32>
    return %2 : tensor<1x128x512xf32>
  }

  "npu.module"() ({
    "npu.kernel"() ({
      "npu.dma_load"() {src = "dram.A", dst = "sram.A_tile"} : () -> ()
      "npu.dma_load"() {src = "dram.B", dst = "sram.B_tile"} : () -> ()
      "npu.matmul_tile"() {m = 64, n = 64, k = 128, acc = "fp32"} : () -> ()
      "npu.apply_bias_relu"() : () -> ()
      "npu.dma_store"() {src = "sram.C_tile", dst = "dram.C"} : () -> ()
      "npu.return"() : () -> ()
    }) : () -> ()
  }) : () -> ()
}
