// Lecture 14 - pseudo before/after for vector.contract tensorization

// BEFORE
#matmul_trait = {
  indexing_maps = [
    affine_map<(m, n, k) -> (m, k)>,
    affine_map<(m, n, k) -> (k, n)>,
    affine_map<(m, n, k) -> (m, n)>
  ],
  iterator_types = ["parallel", "parallel", "reduction"]
}

module {
  func.func @before(%a: vector<16x32xi8>, %b: vector<32x16xi8>, %acc: vector<16x16xi32>) -> vector<16x16xi32> {
    %out = vector.contract #matmul_trait %a, %b, %acc
      : vector<16x32xi8>, vector<32x16xi8> into vector<16x16xi32>
    return %out : vector<16x16xi32>
  }
}

// AFTER - pseudo NPU dialect
module {
  func.func @after(%a: vector<16x32xi8>, %b: vector<32x16xi8>, %acc: vector<16x16xi32>) -> vector<16x16xi32> {
    %out = npu_kernel.matmul %a, %b, %acc
      {m = 16 : i64, n = 16 : i64, k = 32 : i64,
       lhs_layout = "mk", rhs_layout = "kn", acc_type = "i32"}
      : vector<16x32xi8>, vector<32x16xi8>, vector<16x16xi32> -> vector<16x16xi32>
    return %out : vector<16x16xi32>
  }
}
