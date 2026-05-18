// Lecture 14 - Vector type examples
// This file is intended for reading/annotation; some snippets use pseudo target attributes.

module {
  func.func @vector_type_examples(%A: memref<?x?xi8>, %C: memref<?x?xi32>, %m: index, %k: index, %n: index) {
    %c0_i8 = arith.constant 0 : i8
    %c0_i32 = arith.constant 0 : i32

    // A tile: M x K = 16 x 32.
    %a = vector.transfer_read %A[%m, %k], %c0_i8
      {permutation_map = affine_map<(d0, d1) -> (d0, d1)>, in_bounds = [true, true]}
      : memref<?x?xi8>, vector<16x32xi8>

    // Accumulator tile: M x N = 16 x 16.
    %acc = vector.broadcast %c0_i32 : i32 to vector<16x16xi32>

    vector.transfer_write %acc, %C[%m, %n]
      {permutation_map = affine_map<(d0, d1) -> (d0, d1)>, in_bounds = [true, true]}
      : vector<16x16xi32>, memref<?x?xi32>
    return
  }
}
