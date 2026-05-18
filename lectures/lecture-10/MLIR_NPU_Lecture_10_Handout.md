# MLIR NPU Lecture 10 - Rewrite Pattern

## 주제
Rewrite Pattern: MatMul + Add + ReLU fusion pattern 설계

## 학습 목표
- `RewritePattern`, `OpRewritePattern`, `PatternRewriter`의 역할을 구분한다.
- `matchAndRewrite` 안에서 match phase와 rewrite phase를 분리한다.
- greedy rewrite driver의 worklist/fixpoint 동작을 NPU fusion pass와 연결한다.
- MatMul + Bias + ReLU 패턴을 안전하게 fusion하기 위한 semantic, shape, dtype, layout, use-def, target legality 조건을 설계한다.
- FileCheck positive/negative regression test를 작성한다.

## 핵심 개념

### 1. Rewrite Pattern
MLIR pattern rewriting은 크게 pattern definition과 pattern application으로 나뉜다. Pattern definition은 어떤 root operation을 볼 것인지, 어떤 조건을 만족해야 하는지, 어떤 IR로 바꿀 것인지를 정의한다. Pattern application은 driver가 worklist를 돌며 pattern benefit과 cost model에 따라 실제로 pattern을 적용하는 과정이다.

### 2. PatternRewriter 규칙
`matchAndRewrite`에서 IR mutation은 반드시 `PatternRewriter` API를 통해 수행해야 한다. Pattern이 직접 op를 erase하거나 operand를 수정하면 driver의 내부 상태가 깨질 수 있다. 따라서 create, replace, erase, modify-in-place 모두 rewriter를 통해 수행한다.

### 3. NPU Fusion Pattern
MatMul + Add + ReLU fusion은 단순히 op 3개를 하나로 줄이는 변환이 아니다. NPU 관점에서는 다음을 증명해야 한다.
- Bias broadcast가 hardware epilogue에서 표현 가능하다.
- accumulator dtype, requant, activation clamp가 target kernel과 일치한다.
- MatMul/Add/ReLU의 intermediate tensor lifetime 제거가 correctness를 깨지 않는다.
- layout conversion 비용보다 fusion benefit이 크다.

## 추천 실습 순서
1. `01_linalg_matmul_add_relu_before.mlir`에서 MatMul, AddBias, ReLU producer-consumer chain을 표시한다.
2. `03_matmul_add_relu_fusion_pattern.cpp`의 `notifyMatchFailure` 조건을 채운다.
3. `08_fusion_contract_checklist.md`를 기준으로 fusion 가능/불가능 케이스를 분류한다.
4. `06_filecheck_fusion_test.mlir`, `07_negative_no_fusion_test.mlir`에 CHECK line을 보강한다.

## 한 줄 요약
Rewrite pattern은 compiler optimization의 가장 작은 실행 단위이고, NPU compiler에서 pattern의 품질은 target contract를 얼마나 정확히 증명하느냐로 결정된다.
