// RUN: mlir-opt %s --npu-vector-contract-tensorize | FileCheck %s

#matmul_trait = {
  indexing_maps = [
    affine_map<(m, n, k) -> (m, k)>,
    affine_map<(m, n, k) -> (k, n)>,
    affine_map<(m, n, k) -> (m, n)>
  ],
  iterator_types = ["parallel", "parallel", "reduction"]
}

// CHECK-LABEL: func.func @m16n16k32_i8
// CHECK: npu_kernel.matmul
// CHECK-SAME: m = 16
// CHECK-SAME: n = 16
// CHECK-SAME: k = 32
// CHECK-NOT: vector.contract
func.func @m16n16k32_i8(%a: vector<16x32xi8>, %b: vector<32x16xi8>, %acc: vector<16x16xi32>) -> vector<16x16xi32> {
  %out = vector.contract #matmul_trait %a, %b, %acc
    : vector<16x32xi8>, vector<32x16xi8> into vector<16x16xi32>
  return %out : vector<16x16xi32>
}

// CHECK-LABEL: func.func @unsupported_k48
// CHECK-NOT: npu_kernel.matmul
// CHECK: vector.contract
func.func @unsupported_k48(%a: vector<16x48xi8>, %b: vector<48x16xi8>, %acc: vector<16x16xi32>) -> vector<16x16xi32> {
  %out = vector.contract #matmul_trait %a, %b, %acc
    : vector<16x48xi8>, vector<48x16xi8> into vector<16x16xi32>
  return %out : vector<16x16xi32>
}
