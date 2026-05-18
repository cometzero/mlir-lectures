// MLIR_NPU_Lecture_02_IR_Examples.mlir
// Purpose: practice reading Operation, Value, Type, and Attribute.

// Example 1: Structured op style using standard MLIR dialects.
module {
  func.func @matmul_basic(
      %lhs: tensor<4x8xf32>,
      %rhs: tensor<8x16xf32>) -> tensor<4x16xf32> {
    %zero = arith.constant 0.000000e+00 : f32
    %init = tensor.empty() : tensor<4x16xf32>
    %filled = linalg.fill ins(%zero : f32)
              outs(%init : tensor<4x16xf32>) -> tensor<4x16xf32>
    %out = linalg.matmul
        ins(%lhs, %rhs : tensor<4x8xf32>, tensor<8x16xf32>)
        outs(%filled : tensor<4x16xf32>) -> tensor<4x16xf32>
    func.return %out : tensor<4x16xf32>
  }
}

// Example 2: Generic assembly form for an operation.
module {
  func.func @generic_form(%a: f32, %b: f32) -> f32 {
    %0 = "arith.addf"(%a, %b) : (f32, f32) -> f32
    func.return %0 : f32
  }
}

// Example 3: Pseudo NPU operation anatomy. Requires a custom npu dialect to parse.
// %out, %token = "npu.matmul"(%A, %B, %acc) {
//   tile = [16, 16, 64], memory_space = "sram", acc_type = i32
// } : (memref<128x64xf16>, memref<64x128xf16>, memref<128x128xi32>)
//     -> (memref<128x128xi32>, !npu.token) loc("kernel.mlir":12:3)

// Example 4: Attribute-heavy pseudo linalg.generic snippet.
// #map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
// #map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
// #map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
// 
// "linalg.generic"(%A, %B, %C) {
//   indexing_maps = [#map0, #map1, #map2],
//   iterator_types = ["parallel", "parallel", "reduction"]
// } : (tensor<4x8xf32>, tensor<8x16xf32>, tensor<4x16xf32>) -> tensor<4x16xf32>
