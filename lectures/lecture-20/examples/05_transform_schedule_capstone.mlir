// Pseudo Transform Dialect schedule for the capstone.
transform.sequence failures(propagate) {
^bb0(%root: !transform.any_op):
  %matmuls = transform.structured.match ops{["linalg.matmul"]} in %root
    : (!transform.any_op) -> !transform.any_op
  %tiled, %loops = transform.structured.tile_using_for %matmuls
    [16, 16, 32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  %epilogue = transform.structured.match ops{["arith.addf", "math.tanh", "arith.mulf"]} in %root
    : (!transform.any_op) -> !transform.any_op
  transform.structured.fuse_into_containing_op %epilogue into %loops
    : (!transform.any_op, !transform.any_op) -> !transform.any_op
  transform.structured.vectorize %tiled : !transform.any_op
  transform.yield
}
