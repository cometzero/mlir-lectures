// Pseudo example: a later read can prevent safe in-place update.
// Students should mark where a copy/fresh allocation may be required.

func.func @raw_conflict(%input: tensor<64xf32>) -> (tensor<64xf32>, f32) {
  %c0 = arith.constant 0 : index
  %old = tensor.extract %input[%c0] : tensor<64xf32>
  %dst = bufferization.alloc_tensor() : tensor<64xf32>
  %updated = "npu_graph.update"(%input, %dst) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
  // If old input must be read after update, updating input in-place would be incorrect.
  return %updated, %old : tensor<64xf32>, f32
}
