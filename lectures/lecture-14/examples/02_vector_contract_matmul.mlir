// Lecture 14 - vector.contract MatMul example

#matmul_trait = {
  indexing_maps = [
    affine_map<(m, n, k) -> (m, k)>,
    affine_map<(m, n, k) -> (k, n)>,
    affine_map<(m, n, k) -> (m, n)>
  ],
  iterator_types = ["parallel", "parallel", "reduction"]
}

module {
  func.func @m16n16k32_i8(%A: memref<?x?xi8>, %B: memref<?x?xi8>, %C: memref<?x?xi32>,
                          %m: index, %n: index, %k: index) {
    %c0_i8 = arith.constant 0 : i8
    %c0_i32 = arith.constant 0 : i32
    %a = vector.transfer_read %A[%m, %k], %c0_i8
      {in_bounds = [true, true]} : memref<?x?xi8>, vector<16x32xi8>
    %b = vector.transfer_read %B[%k, %n], %c0_i8
      {in_bounds = [true, true]} : memref<?x?xi8>, vector<32x16xi8>
    %acc = vector.broadcast %c0_i32 : i32 to vector<16x16xi32>
    %out = vector.contract #matmul_trait %a, %b, %acc
      : vector<16x32xi8>, vector<32x16xi8> into vector<16x16xi32>
    vector.transfer_write %out, %C[%m, %n]
      {in_bounds = [true, true]} : vector<16x16xi32>, memref<?x?xi32>
    return
  }
}
