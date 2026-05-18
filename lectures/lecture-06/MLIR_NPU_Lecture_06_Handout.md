# MLIR NPU Compiler - Lecture 06 Handout

## Topic

StableHLO, TOSA, and AI model input IR

## 핵심 메시지

StableHLO와 TOSA는 NPU의 최종 실행 스케줄이 아니라, framework에서 넘어온 model semantics를 compiler가 안전하게 해석하기 위한 입력 IR입니다. NPU compiler는 이 입력 IR을 기준으로 op legality, shape, dtype, layout, quantization, numerical approximation을 audit한 뒤 Linalg/Tensor 또는 custom NPU dialect로 lowering합니다.

## StableHLO vs TOSA

| 관점 | StableHLO | TOSA |
|---|---|---|
| 주요 역할 | ML framework와 ML compiler 사이 portability layer | Tensor-level operator set 및 backend-friendly normalization |
| 추상화 수준 | high-level HLO semantics | tensor-level operation semantics |
| NPU compiler 포인트 | dot_general, convolution, reduce, broadcast, dynamic shape semantics | matmul, conv2d, reduce_sum, rsqrt, rescale, table, elementwise ops |
| 장점 | framework semantics 보존 | hardware/backend concern과 minimal op set 지향 |
| 주의점 | 너무 늦게 decompose하면 target legality 판단이 늦어짐 | 너무 일찍 decompose하면 fusion/pattern intent가 사라질 수 있음 |

## LayerNorm decomposition

```text
mean = reduce_sum(x, axis=-1) / K
center = x - broadcast(mean)
var = reduce_sum(center * center, axis=-1) / K
inv = rsqrt(var + eps)
y = center * broadcast(inv)
out = y * gamma + beta
```

## GELU decomposition

Exact:

```text
0.5 * x * (1 + erf(x / sqrt(2)))
```

Approx:

```text
0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
```

## 실습 산출물

1. StableHLO 또는 TOSA op list에서 NPU supported/unsupported op audit table 작성
2. LayerNorm primitive graph 작성
3. GELU exact vs approximate lowering 선택 및 error policy 작성
4. `MatMul + Bias + GELU` fusion candidate 분석
