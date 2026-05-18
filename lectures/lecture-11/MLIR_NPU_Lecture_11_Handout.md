# Lecture 11 - Dialect Conversion

## 핵심 메시지
Dialect Conversion은 NPU target capability를 compiler pass의 legality contract로 만드는 단계입니다. 단순히 op를 바꾸는 것이 아니라, 이 boundary 이후 어떤 op/type/layout/dtype이 허용되는지 명시합니다.

## Framework
- `ConversionTarget`: legal/illegal/dynamic legal op set 정의
- `ConversionPattern`: illegal op를 legal op sequence로 rewrite
- `TypeConverter`: type, block argument, materialization conversion
- `applyPartialConversion`: progressive lowering
- `applyFullConversion`: final boundary
- `applyAnalysisConversion`: legalizable 여부 분석

## NPU Legal Op Set 작성법
Legal op set은 다음을 함께 포함해야 합니다.

```text
op name
+ shape policy
+ dtype/accumulator policy
+ layout/packing policy
+ quantization policy
+ memory/SRAM constraint
+ approximation/fallback policy
```

## Unsupported Op Lowering 전략
1. Direct lower: target op 하나로 대응
2. Decompose: legal primitive graph로 분해
3. Approximate: LUT/polynomial/Newton 등으로 근사
4. Fallback: CPU/DSP/runtime op로 명시
5. Reject: diagnostic과 함께 compile failure

## NPU Compiler 관점의 가장 흔한 실수
- entire dialect를 너무 빨리 legal로 선언
- dynamic legality 없이 op name만 보고 legal 처리
- ConversionPattern에서 adaptor operand 대신 original operand 사용
- approximation lowering에 numerical tolerance test 없음
- unsupported shape/dtype/layout diagnostic이 모호함

## 실습 산출물
- `01_npu_legal_op_set.yaml`
- `04_conversion_target_skeleton.cpp`
- `05_conversion_pattern_skeleton.cpp`
- `07_filecheck_conversion_test.mlir`
- `09_unsupported_op_lowering_matrix.md`
