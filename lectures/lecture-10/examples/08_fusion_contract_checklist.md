# Lecture 10 - Fusion Contract Checklist

## Pattern identity
- Root op: ReLU-like elementwise op or linalg.generic with max(x, 0)
- Producer 1: AddBias-like elementwise op
- Producer 2: MatMul-like contraction op
- Replacement: npu_graph.matmul_epilogue

## Semantic checks
- Add input classification is unambiguous: exactly one MatMul result and one bias tensor/scalar.
- ReLU constant is zero and compare/max semantics match the target activation.
- No operation has side effects that would be reordered by fusion.

## Use-def checks
- MatMul result has one use, unless the pass intentionally clones or keeps the original result.
- Add result has one use.
- Fused output type equals original ReLU output type.

## Shape/layout checks
- Bias broadcast is supported by the NPU kernel.
- Output rank and major dimensions are supported by the target.
- Required layout conversion cost does not exceed the benefit of fusion.

## Quantization checks
- Accumulator type is supported.
- Requant scale/zero-point/saturation policy can be represented.
- Activation clamp range is compatible with post-op quantization.

## Test checks
- Positive FileCheck: fused op appears and old ops disappear.
- Negative FileCheck: shared-producer case remains unfused.
- Diagnostic/debug path: notifyMatchFailure explains rejection reason.
