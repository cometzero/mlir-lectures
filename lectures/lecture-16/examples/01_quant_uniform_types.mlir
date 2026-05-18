// Lecture 16 - Quant uniform type examples.
// These examples are pseudo-MLIR intended for reading and annotation.

func.func @quant_type_examples(%a: tensor<1x128xf32>, %w: tensor<128x64xf32>) {
  // Per-tensor activation quantization: one scale and one zero point.
  %qa = quant.qcast %a : tensor<1x128xf32>
      to tensor<1x128x!quant.uniform<i8:f32, 0.03125:-3>>

  // Per-output-channel weight quantization.
  // Channel axis = 1 because shape is [K, N].
  %qw = quant.qcast %w : tensor<128x64xf32>
      to tensor<128x64x!quant.uniform<i8:f32:1,
        {0.015, 0.017, 0.022, 0.019}>>

  // Expose storage type for integer-only lowering.
  %a_i8 = quant.scast %qa : tensor<1x128x!quant.uniform<i8:f32, 0.03125:-3>>
      to tensor<1x128xi8>
  %w_i8 = quant.scast %qw : tensor<128x64x!quant.uniform<i8:f32:1,
        {0.015, 0.017, 0.022, 0.019}>> to tensor<128x64xi8>
  return
}
