// Transform IR schedule sketch.
// Syntax is close to upstream MLIR but may need adjustment for a specific LLVM/MLIR revision.
// Check upstream Transform Dialect docs and examples/transform tests for exact current spelling.

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op)
      attributes {transform.readonly} {
    // 1. Select the MatMul payload operation.
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %root
      : (!transform.any_op) -> !transform.any_op

    // 2. Tile M/N/K to the NPU native tile shape.
    // M=16, N=16, K=32 for INT8xINT8->INT32 accumulator.
    %tiled_matmul, %loops:3 = transform.structured.tile_using_for %matmul
      tile_sizes [16, 16, 32]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    // 3. Match epilogue op and fuse it into the containing loop/tile when legal.
    %epilogue = transform.structured.match ops{["npu_graph.bias_relu"]} in %root
      : (!transform.any_op) -> !transform.any_op
    // Pseudo: exact fusion target depends on the generated loop/tile handles.
    // %fused, %new_loop = transform.structured.fuse_into_containing_op %epilogue into %loops#0
    //   : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // 4. Vectorize the tiled MatMul. It can later be tensorized to npu_kernel.matmul.
    transform.structured.vectorize %tiled_matmul vector_sizes [16, 16, 32]
      : !transform.any_op

    transform.yield
  }
}
