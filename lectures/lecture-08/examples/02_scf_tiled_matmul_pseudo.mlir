// Lecture 08 - Example 02: SCF pseudo tiled matmul.
// Pseudo npu.* ops are intentionally used for schedule annotation.

func.func @scf_tiled_matmul_pseudo(
    %A: memref<128x256xi8>,
    %B: memref<256x128xi8>,
    %C: memref<128x128xi32>) {
  %c0 = arith.constant 0 : index
  %M = arith.constant 128 : index
  %N = arith.constant 128 : index
  %K = arith.constant 256 : index
  %TM = arith.constant 32 : index
  %TN = arith.constant 64 : index
  %TK = arith.constant 64 : index

  scf.for %m0 = %c0 to %M step %TM {
    scf.for %n0 = %c0 to %N step %TN {
      "npu.acc.zero"(%m0, %n0, %TM, %TN) : (index, index, index, index) -> ()
      scf.for %k0 = %c0 to %K step %TK {
        "npu.dma.load_a"(%A, %m0, %k0, %TM, %TK) : (memref<128x256xi8>, index, index, index, index) -> ()
        "npu.dma.load_b"(%B, %k0, %n0, %TK, %TN) : (memref<256x128xi8>, index, index, index, index) -> ()
        "npu.barrier.wait_dma"() : () -> ()
        "npu.matmul.tile"(%m0, %n0, %k0, %TM, %TN, %TK) : (index, index, index, index, index, index) -> ()
      }
      "npu.dma.store_c"(%C, %m0, %n0, %TM, %TN) : (memref<128x128xi32>, index, index, index, index) -> ()
    }
  }
  return
}
