// Lecture 17: transform dialect schedule sketch for MLP fusion and layout.
// Pseudo Transform IR: names may differ across MLIR revisions.

transform.sequence failures(propagate) {
^bb1(%module: !transform.any_op):
  %matmuls = transform.structured.match ops{["linalg.matmul"]} in %module : (!transform.any_op) -> !transform.any_op
  %mlp = transform.npu.match_mlp_pattern %matmuls {
    activation = "gelu", require_single_use = true
  } : (!transform.any_op) -> !transform.any_op

  %fused = transform.npu.fuse_mlp_epilogue %mlp {
    epilogue = "bias_gelu_tanh",
    output_layout = "MxN_blocked_16x16"
  } : (!transform.any_op) -> !transform.any_op

  transform.npu.propagate_layout %fused {
    through = ["elementwise", "reshape_view"],
    materialize_at = ["function_boundary", "row_major_only_op"]
  } : (!transform.any_op) -> ()
}
