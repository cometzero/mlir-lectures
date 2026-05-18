// Lecture 08 - Example 03: affine tiled matmul skeleton.

func.func @affine_tiled_matmul(
    %A: memref<128x256xf32>,
    %B: memref<256x128xf32>,
    %C: memref<128x128xf32>) {
  affine.for %m0 = 0 to 128 step 32 {
    affine.for %n0 = 0 to 128 step 64 {
      affine.for %k0 = 0 to 256 step 64 {
        affine.for %mi = 0 to 32 {
          affine.for %ni = 0 to 64 {
            affine.for %ki = 0 to 64 {
              %a = affine.load %A[%m0 + %mi, %k0 + %ki] : memref<128x256xf32>
              %b = affine.load %B[%k0 + %ki, %n0 + %ni] : memref<256x128xf32>
              %c = affine.load %C[%m0 + %mi, %n0 + %ni] : memref<128x128xf32>
              %p = arith.mulf %a, %b : f32
              %s = arith.addf %c, %p : f32
              affine.store %s, %C[%m0 + %mi, %n0 + %ni] : memref<128x128xf32>
            }
          }
        }
      }
    }
  }
  return
}
