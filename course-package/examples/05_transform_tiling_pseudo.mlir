// Transform dialect pseudo example. Adjust op names to your LLVM/MLIR revision.
transform.sequence failures(propagate) {
^bb0(%root: !transform.any_op):
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %root
    : (!transform.any_op) -> !transform.any_op
  %tiled, %loops = transform.structured.tile_using_for %matmul [16, 16, 64]
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  transform.yield
}
