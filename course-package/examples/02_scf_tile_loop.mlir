// M/N/K tiled loop skeleton.
module {
  func.func @tile_loop(%M: index, %N: index, %K: index) {
    %c0 = arith.constant 0 : index
    %TM = arith.constant 16 : index
    %TN = arith.constant 16 : index
    %TK = arith.constant 64 : index
    scf.for %m = %c0 to %M step %TM {
      scf.for %n = %c0 to %N step %TN {
        scf.for %k = %c0 to %K step %TK {
          // dma A/B tile -> compute -> accumulate
        }
      }
    }
    return
  }
}
