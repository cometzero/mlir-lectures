# MLIR NPU Lecture 16 Quiz

1. `!quant.uniform<i8:f32, 0.25:-2>`에서 stored type, expressed type, scale, zero_point를 쓰세요.
2. 왜 zero_point는 storage range 안에 있어야 합니까?
3. `quant.qcast`, `quant.dcast`, `quant.scast`의 차이를 설명하세요.
4. INT8 MatMul에서 accumulator가 보통 i32인 이유를 worst-case product 관점에서 설명하세요.
5. Per-channel weight quantization에서 channel axis가 layout transform 이후 바뀌면 어떤 문제가 생깁니까?
6. `scale_a=0.02`, `scale_w=0.03`, `scale_y=0.01`일 때 requant ratio r을 계산하세요.
7. `tosa.rescale`이 floating multiply 대신 어떤 integer 연산으로 requantization을 표현하는지 설명하세요.
8. INT4 weight packing에서 nibble order를 ABI에 명시해야 하는 이유는 무엇입니까?
9. Bias scale이 `scale_a * scale_w[channel]`와 다르면 어떤 문제가 발생합니까?
10. NPU compiler의 quantization negative test 세 가지를 제안하세요.

## 정답 요약

1. stored i8, expressed f32, scale 0.25, zp -2.
2. real zero를 정확히 표현해야 padding/zero 값이 bias를 만들지 않기 때문.
3. qcast는 quantize, dcast는 dequantize, scast는 storage reinterpretation.
4. K번의 integer product 누적이 overflow하지 않도록 충분한 폭이 필요.
5. 잘못된 scale이 channel에 적용되어 output별 scaling이 틀어짐.
6. r = 0.02 * 0.03 / 0.01 = 0.06.
7. multiplier, add/zero point, shift, rounding, clamp.
8. 같은 byte도 high/low 해석이 다르면 weight 값이 달라짐.
9. bias가 잘못 스케일되어 output에 systematic error가 생김.
10. scale length mismatch, K misalignment, overflow, unsupported rounding, invalid zp 등.
