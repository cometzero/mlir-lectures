# Lecture 17 Quiz - Fusion and Layout Propagation

1. `MatMul + Bias + GELU` fusion이 항상 좋은 최적화가 아닌 이유를 3가지 쓰세요.
2. Layout propagation과 layout materialization의 차이를 설명하세요.
3. Transformer MLP에서 first GEMM output의 layout을 blocked로 유지하면 second GEMM에 어떤 장단점이 있나요?
4. Per-channel quantization axis가 layout propagation에서 문제가 되는 사례를 쓰세요.
5. 다음 중 FileCheck에 반드시 넣어야 할 항목은 무엇인가요? `fused op 존재`, `unfused op 부재`, `layout attr`, `materialization op count`.
6. SwiGLU fusion에서 SRAM live range가 증가하는 이유를 설명하세요.
7. Cost model에 compute cycles만 넣으면 안 되는 이유를 설명하세요.
8. Function boundary에서 row-major ABI를 요구할 때 compiler가 삽입해야 하는 op는 무엇인가요?

## 정답 힌트

- SRAM, live range, layout repack, unsupported activation approximation, multiple consumers, quant axis mismatch를 생각하세요.
