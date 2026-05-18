# MLIR NPU Compiler - Lecture 02 Handout

**Topic:** MLIR IR Basic Structure - Operation / Value / Type / Attribute  
**Output:** Annotated IR + NPU compiler IR audit checklist

## Reading Order
1. Operation name: dialect and abstraction level.
2. Results and operands: SSA values produced and consumed.
3. Type signature: shape, element type, memory space, vector/token information.
4. Attributes: compile-time metadata such as tile, layout, iterator type, quantization scale.
5. Location: debug/profiling provenance.

## Operation Anatomy
```mlir
%out, %token = "npu.matmul"(%A, %B, %acc) {
  tile = [16, 16, 64], memory_space = "sram", acc_type = i32
} : (memref<128x64xf16>, memref<64x128xf16>, memref<128x128xi32>)
    -> (memref<128x128xi32>, !npu.token) loc("kernel.mlir":12:3)
```

## Minimal Example
```mlir
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
```

## NPU Compiler Audit Checklist
- Is the operation at the right abstraction level for the current pass?
- Are def-use chains preserved after rewrite?
- Does each type encode the information needed by the next lowering stage?
- Are dynamic runtime quantities modeled as operands/descriptors rather than static attributes?
- Are source locations preserved through lowering?
- Are NPU legality constraints explicitly verifiable?

## References
- MLIR Language Reference
- Understanding the IR Structure
- Toy Tutorial Chapter 2: Emitting Basic MLIR
- Builtin Dialect
- Func Dialect
- Linalg Dialect
