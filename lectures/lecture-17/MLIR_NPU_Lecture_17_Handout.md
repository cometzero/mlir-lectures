# MLIR/NPU Compiler Lecture 17 - Fusion and Layout Propagation

## 목표

17강의 목표는 Transformer MLP를 예제로 사용해 fusion 후보를 찾고, layout propagation과 cost model을 결합해 NPU에서 실제로 이득이 나는 fusion만 선택하는 방법을 익히는 것입니다.

## 핵심 개념

- Fusion은 pattern matching이 아니라 legality + cost + layout contract입니다.
- Layout propagation은 physical layout을 elementwise/view-like op를 통해 밀고 나가며, materialization/repack은 explicit boundary에서만 삽입하는 방식입니다.
- Transformer MLP의 대표 후보는 `MatMul + Bias + GELU`, `MatMul + Bias`, SwiGLU의 `SiLU + Mul`, weight prepack, activation layout propagation입니다.
- 좋은 fusion은 DRAM traffic을 줄이지만, 나쁜 fusion은 live range 증가, SRAM overflow, bank conflict, fallback materialization으로 성능을 떨어뜨립니다.

## NPU 관점 Checklist

1. single-use 또는 reuse 이득이 명확한가?
2. shape/dtype/layout/quantization이 target legal op set에 맞는가?
3. SRAM/ACCUM footprint가 예산 안에 들어가는가?
4. output layout이 다음 consumer의 preferred layout과 맞는가?
5. materialization point가 명시되어 FileCheck로 테스트 가능한가?
6. numerical behavior, activation approximation, quant clamp가 문서화되어 있는가?

## 실습 산출물

- Transformer MLP fusion candidate matrix
- layout propagation graph
- fused NPU graph pseudo IR
- memory traffic calculator 결과
- FileCheck positive/negative test skeleton
