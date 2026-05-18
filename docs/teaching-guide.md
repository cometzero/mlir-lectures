# Teaching Guide

## 강의 운영 원칙

- 슬라이드 중심이 아니라 IR 중심으로 진행합니다. 슬라이드는 개념 도입, `examples/`는 이해 검증, `Worksheet`는 산출물 작성에 사용합니다.
- 모든 lecture에서 "무엇을 operand로 둘 것인가, 무엇을 attribute/type으로 둘 것인가"를 반복적으로 묻습니다.
- NPU compiler 맥락에서는 correctness와 performance를 분리합니다. verifier가 reject할 조건과 cost model이 판단할 조건을 구분하게 합니다.
- pseudo dialect 예제가 실패하면 실패 이유를 수업 소재로 사용합니다. 실제 compiler에서는 dialect registration, parser/printer, verifier, pass pipeline이 필요하다는 점을 연결합니다.

## 90~120분 템플릿

1. Recap: 이전 lecture 산출물 확인 (5~10분)
2. Concept: Handout/Slides 핵심 설명 (25~35분)
3. IR Walkthrough: examples 파일 함께 읽기 (20~30분)
4. Lab: Worksheet 또는 skeleton 수정 (25~35분)
5. Review: Quiz와 negative case 토론 (10분)

## 실습 피드백 기준

- IR 구조를 정확히 읽었는가?
- target constraint(shape, dtype, layout, quant, memory, ABI)를 명시했는가?
- success case와 failure case를 모두 생각했는가?
- pass boundary와 runtime boundary를 혼동하지 않았는가?
- 산출물이 다음 lecture에서 재사용 가능하게 정리되었는가?

## Capstone 연결 방법

Lecture 20 전에 각 lecture의 산출물을 다음 bucket으로 모읍니다.

- input IR contract: 01~07
- scheduling/fusion/layout/quant: 08, 10, 14~17
- legal op set/conversion/bufferization/memory: 09, 11~13
- custom dialect/runtime ABI: 18~19
- validation/cost model/rubric: 04, 16~20
