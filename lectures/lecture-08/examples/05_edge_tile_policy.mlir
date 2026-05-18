// Lecture 08 - Example 05: edge tile policy pseudo IR.

func.func @edge_tile_policy(%M: index, %N: index) {
  %c0 = arith.constant 0 : index
  %TM = arith.constant 32 : index
  %TN = arith.constant 64 : index

  scf.for %m0 = %c0 to %M step %TM {
    scf.for %n0 = %c0 to %N step %TN {
      // Real compiler policy:
      // tm_eff = min(TM, M - m0)
      // tn_eff = min(TN, N - n0)
      // choose mask, padding, or fallback micro-kernel.
      "npu.tile.boundary"(%m0, %n0, %M, %N) : (index, index, index, index) -> ()
    }
  }
  return
}
