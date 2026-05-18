# MLIR NPU Compiler - Lecture 06 Quiz

## Questions

1. StableHLO가 NPU compiler input IR로 유용한 이유를 한 문장으로 설명하세요.
2. TOSA가 hardware target concern을 고려한다는 말이 NPU compiler 설계에서 어떤 의미인지 설명하세요.
3. LayerNorm을 primitive ops로 분해할 때 필요한 reduction은 몇 번인가요?
4. LayerNorm에서 epsilon은 어느 단계에 들어가야 하나요?
5. GELU exact 수식과 tanh approximation 수식의 차이를 설명하세요.
6. GELU를 MatMul epilogue에 fuse하기 전에 확인해야 할 조건 3가지를 쓰세요.
7. `stablehlo.broadcast_in_dim` 또는 `tosa.tile`이 NPU compiler에서 중요한 이유는 무엇인가요?
8. Input IR audit matrix에 반드시 포함해야 하는 항목 5가지를 쓰세요.
9. TOSA `rescale` 또는 `table` 류 operation이 quantized NPU path에서 중요한 이유를 설명하세요.
10. StableHLO/TOSA에서 바로 command buffer를 만들면 위험한 이유를 설명하세요.

## Suggested Answers

1. Framework별 model semantics를 compiler가 공통 형식으로 받아 target lowering을 시작할 수 있게 해준다.
2. Operator set과 semantic boundary가 backend kernel/codegen 설계와 연결되어 NPU legal op set과 unsupported op decomposition을 체계화할 수 있다는 뜻이다.
3. 평균과 분산 계산에 각각 1번씩 총 2번의 reduction이 필요하다.
4. variance에 더한 뒤 rsqrt를 적용해야 한다.
5. Exact는 erf를 쓰고, approximation은 tanh와 cubic polynomial term을 사용한다.
6. dtype/accumulator precision, approximation error tolerance, post-op order, layout, quantization scale 등.
7. Logical broadcast semantics가 실제 memory layout과 vectorization/bufferization에 영향을 주기 때문이다.
8. op semantics, shape, dtype, layout, reduction axis, broadcast, numerical policy, fusion, quantization 등.
9. accumulator 결과를 output scale/zero-point에 맞춰 requantize하거나 LUT activation으로 근사할 수 있기 때문이다.
10. shape/layout/fusion/quantization/legalization 검증이 생략되어 correctness와 performance 문제가 생길 수 있다.
