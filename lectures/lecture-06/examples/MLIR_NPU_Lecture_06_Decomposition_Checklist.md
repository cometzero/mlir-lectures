# MLIR NPU Lecture 06 - Decomposition Checklist

## Input IR audit

| Item | Question | NPU compiler decision |
|---|---|---|
| Op semantics | Primitive or composite pattern? | Preserve, decompose, fuse, or fallback |
| Shape | Static rank and key dimensions? | Specialize, guard, or fallback |
| DType | f32/f16/bf16/int8/int4? | Select accumulator and conversion rules |
| Layout | Logical only or physical requirement? | Apply layout propagation before bufferization |
| Reduction axis | Last dim, channel dim, or dynamic? | Map to reduction kernel or decompose loops |
| Broadcast | Explicit and legal? | Materialize, vector-load, or rewrite |
| Numerical behavior | epsilon, rounding, approximation? | Add verifier constraints and golden tests |
| Fusion | Bandwidth/SRAM benefit? | Fuse or keep separate based on reuse |
| Quantization | Scale/zero-point granularity? | Insert rescale/requant or preserve metadata |

## LayerNorm target

```text
mean = reduce_sum(x, axis=-1) / K
center = x - broadcast(mean)
var = reduce_sum(center * center, axis=-1) / K
inv = rsqrt(var + eps)
out = center * broadcast(inv) * gamma + beta
```

## GELU target

```text
Exact: 0.5 * x * (1 + erf(x / sqrt(2)))
Approx: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
```

NPU choices: exact erf, tanh approximation, sigmoid/polynomial approximation, LUT/table, or MatMul epilogue fusion.
