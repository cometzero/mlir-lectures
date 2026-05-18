// Lecture 14 - vector.transfer_read / transfer_write tile movement

module {
  func.func @tile_move(%A: memref<?x?xf16, #gpu.address_space<workgroup>>, %C: memref<?x?xf16>,
                       %m: index, %n: index) {
    %pad = arith.constant 0.0 : f16
    %tile = vector.transfer_read %A[%m, %n], %pad
      {permutation_map = affine_map<(d0, d1) -> (d0, d1)>, in_bounds = [true, true]}
      : memref<?x?xf16, #gpu.address_space<workgroup>>, vector<16x16xf16>
    vector.transfer_write %tile, %C[%m, %n]
      {permutation_map = affine_map<(d0, d1) -> (d0, d1)>, in_bounds = [true, true]}
      : vector<16x16xf16>, memref<?x?xf16>
    return
  }
}
