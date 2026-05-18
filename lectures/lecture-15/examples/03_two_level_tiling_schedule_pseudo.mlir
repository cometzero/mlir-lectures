// Two-level tiling pseudo schedule for edge NPU.
// L1: outer tiles assigned to cores/clusters.
// L0: inner tiles mapped to native tensor/matrix instruction.

transform.named_sequence @npu_matmul_two_level(%root: !transform.any_op)
    attributes {transform.readonly} {
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %root
    : (!transform.any_op) -> !transform.any_op

  // Outer tiling for SRAM residency and parallelism.
  %outer_tiled, %outer_loops:3 = transform.structured.tile_using_for %matmul
    tile_sizes [64, 64, 64]
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

  // Inner tiling for native NPU array.
  %inner_tiled, %inner_loops:3 = transform.structured.tile_using_for %outer_tiled
    tile_sizes [16, 16, 32]
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

  transform.structured.vectorize %inner_tiled vector_sizes [16, 16, 32]
    : !transform.any_op
  transform.yield
}
