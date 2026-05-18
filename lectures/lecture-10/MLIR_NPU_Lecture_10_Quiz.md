# MLIR NPU Lecture 10 - Quiz

## 객관식

1. `PatternRewriter`를 사용하는 가장 중요한 이유는?
   - A. code generation 속도를 높이기 위해
   - B. pattern driver가 IR mutation 상태를 추적할 수 있게 하기 위해
   - C. 모든 op를 LLVM IR로 바로 변환하기 위해
   - D. FileCheck를 자동 생성하기 위해

2. `matchAndRewrite`에서 가장 안전한 순서는?
   - A. op 생성 -> 조건 확인 -> 실패 시 삭제
   - B. 모든 조건 확인 -> rewrite 수행 -> success 반환
   - C. 조건 일부 확인 -> in-place 수정 -> 나머지 조건 확인
   - D. erase 먼저 수행 -> 새 op 생성

3. MatMul + Bias + ReLU fusion에서 negative test로 가장 중요한 케이스는?
   - A. 모든 producer가 one-use인 케이스
   - B. ReLU constant가 0인 케이스
   - C. MatMul 결과가 다른 op에서도 사용되는 케이스
   - D. output shape이 static인 케이스

## 단답형

4. `notifyMatchFailure`가 NPU compiler debug에서 유용한 이유를 설명하세요.

5. PatternBenefit을 너무 높게 설정하면 어떤 문제가 생길 수 있나요?

6. canonicalization pattern과 target-specific optimization pattern의 차이를 설명하세요.

## 설계 문제

7. 다음 조건의 MatMul + Add + ReLU graph를 fusion해도 되는지 판단하세요.
   - MatMul output: tensor<128x512xi32>
   - Bias: tensor<512xi32>
   - Add output은 ReLU에서만 사용됨
   - ReLU output dtype: tensor<128x512xi8>
   - target epilogue는 int32 accumulator + per-channel requant + relu clamp 지원

8. `FuseMatMulBiasReluPattern`의 match checklist를 6단계 이상 작성하세요.

## 정답 가이드

1. B
2. B
3. C
4. 실패 이유를 pattern debug log와 test reduction에 남겨서, 왜 fusion되지 않았는지 빠르게 추적할 수 있다.
5. 더 정확하거나 더 안전한 pattern보다 먼저 적용되어 불안정한 IR, missed optimization, cyclic rewrite를 유발할 수 있다.
6. canonicalization은 일반적으로 IR을 단순하고 안정적인 표준 형태로 만드는 best-effort 변환이고, target-specific optimization은 특정 NPU의 kernel/layout/quantization/legal op contract를 만족시키는 성능 지향 변환이다.
7. 조건만 보면 fusion 가능 후보이다. 다만 requant scale/zero-point, bias broadcast, activation clamp range, output layout까지 확인해야 최종 허용할 수 있다.
8. 예: root ReLU 확인, producer Add 확인, Add operand classification, MatMul producer 확인, one-use 확인, shape/broadcast 확인, dtype/quant 확인, layout 확인, target epilogue support 확인, replacement type 확인.
