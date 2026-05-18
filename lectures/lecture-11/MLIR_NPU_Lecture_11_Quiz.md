# Lecture 11 Quiz - Dialect Conversion

1. `ConversionTarget`과 일반 `RewritePattern`만 사용하는 pass의 차이를 설명하세요.
2. NPU compiler에서 `addDynamicallyLegalOp`가 필요한 이유를 MatMul 예로 설명하세요.
3. `applyPartialConversion`과 `applyFullConversion`을 각각 어느 pipeline boundary에 쓰는 것이 적절한가요?
4. ConversionPattern에서 adaptor operand를 써야 하는 이유는 무엇인가요?
5. GELU exact op가 NPU에서 unsupported일 때 가능한 lowering 전략 3가지를 쓰세요.
6. TypeConverter가 block argument conversion에 중요한 이유를 설명하세요.
7. unsupported gather op에 대해 CPU fallback을 선택해야 하는 조건을 제시하세요.
8. FileCheck positive test와 negative diagnostic test의 목적 차이를 설명하세요.

## 정답 가이드
1. Dialect Conversion은 target legality를 pass 성공 조건으로 삼는다.
2. Shape/dtype/layout/epilogue/SRAM 조건이 hardware support를 결정하기 때문이다.
3. Partial은 progressive lowering, Full은 final legal boundary에 적합하다.
4. TypeConverter가 remap한 legal operand를 사용해야 verifier failure를 피한다.
5. LUT/polynomial approximation, primitive decomposition, CPU fallback/reject 등.
