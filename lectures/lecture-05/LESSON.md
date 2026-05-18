# Lecture 05 - Dialect System

이 파일은 원본 자료를 강의자가 바로 진행할 수 있도록 재구성한 진행안입니다. 원본 PDF/PPTX/DOCX는 보존되어 있고, 실습은 `examples/`와 `Worksheet`를 기준으로 진행합니다.

## 1. 학습 목표

- 이번 장의 핵심 주제인 `Dialect System`를 MLIR/NPU compiler pipeline 안에서 설명할 수 있습니다.
- 제공된 예제 IR, skeleton, 정책 파일을 읽고 어떤 pass 또는 lowering 단계와 연결되는지 추적할 수 있습니다.
- 강의 후 산출물(worksheet, checklist, pseudo IR, FileCheck test 또는 설계 표)을 작성할 수 있습니다.

## 2. 권장 강의 흐름 (90~120분)

1. 도입 (10분)
   - 이전 lecture의 산출물과 이번 lecture의 위치를 연결합니다.
   - `README.md`의 파일 목록을 보여주고, 오늘 사용할 `Handout`, `examples/`, `Worksheet`를 안내합니다.
2. 핵심 개념 설명 (25~35분)
   - 아래 "강의 포인트" 순서대로 슬라이드와 핸드아웃을 설명합니다.
   - MLIR textual IR에서 operand, attribute, type, region, pass boundary를 직접 가리키며 설명합니다.
3. 예제 Walkthrough (20~30분)
   - `examples/`의 파일을 1개씩 열어 어떤 문제를 모델링하는지 설명합니다.
   - 가능하면 `mlir-opt`, `FileCheck`, Python 스크립트 또는 shell script로 즉석 실행합니다.
4. 실습 (25~35분)
   - 수강자는 worksheet 또는 examples의 skeleton을 채웁니다.
   - 강의자는 legality, verifier, memory/layout/quant/runtime ABI 관점에서 피드백합니다.
5. 정리 및 퀴즈 (10분)
   - `Quiz`의 질문으로 이해도를 점검하고 다음 lecture로 이어지는 unresolved question을 남깁니다.

## 3. 자료 읽는 순서

- 슬라이드: `MLIR_NPU_Lecture_05_Slides.pdf`
- 강의 노트: `MLIR_NPU_Lecture_05_Notes.pdf`
- 핸드아웃: `MLIR_NPU_Lecture_05_Handout.md`
- 워크시트: `MLIR_NPU_Lecture_05_Worksheet.pdf`
- 퀴즈: `MLIR_NPU_Lecture_05_Quiz.md`

## 4. 강의 포인트

- Topic
- Core idea
- Dialect anatomy
- Design rules for NPU compiler
- Example boundary
- Checklist for custom NPU dialect
- References

## 5. 실습 파일

- `examples/MLIR_NPU_Lecture_05_Dialect_Examples.mlir` - MLIR IR 예제
- `examples/MLIR_NPU_Lecture_05_NPU_Dialect_TD.td` - C++/ODS skeleton
- `examples/MLIR_NPU_Lecture_05_Pass_Dependent_Dialects.cpp` - C++/ODS skeleton

## 6. 실행 방법

공통 환경은 루트의 `docs/lab-environment.md`를 먼저 확인하세요.

```bash
# 도구가 설치되어 있다면, 예제 파일을 하나씩 열고 parse/verify를 시도합니다.
# custom/pseudo dialect가 포함된 예제는 실패할 수 있으므로 --allow-unregistered-dialect를 함께 사용합니다.
cd lectures/lecture-05
../../scripts/run_lesson.sh 05 --try-mlir
```


주의: 일부 파일은 교육용 pseudo dialect 또는 skeleton 코드입니다. 실제 MLIR build에 custom dialect가 등록되어 있지 않으면 parse/verify가 실패할 수 있습니다. 실패 자체를 오류로만 보지 말고, 어떤 dialect/type/op registration이 필요한지 토론 자료로 사용하세요.

## 7. 수강자 산출물

- 핵심 IR 또는 설계 요소에 주석을 단 annotated example
- worksheet 답안 또는 checklist
- positive/negative regression test 초안
- NPU target legality, layout, memory, quantization 또는 runtime ABI 관점의 리뷰 메모

## 8. Review 질문

- MLIR Dialect가 정의할 수 있는 대표 구성 요소 3가지를 쓰세요.
- `npu_graph` dialect와 `npu_kernel` dialect를 분리하는 이유를 쓰세요.
- `tile_shape=[16,16,32]`는 operand, attribute, type 중 어디에 두는 것이 자연스럽습니까? 이유는 무엇입니까?
- `npu_rt.launch`가 frontend dialect에 있으면 왜 좋지 않습니까?
- ODS/TableGen을 사용하는 이유를 2가지 쓰세요.

## 9. 강의자 체크리스트

- [ ] 수강자가 `operand / attribute / type / region / pass boundary`를 구분해서 설명할 수 있다.
- [ ] 실습 산출물이 NPU target constraint를 최소 1개 이상 명시한다.
- [ ] 예제의 실패/성공 기준이 FileCheck, verifier, checklist 또는 산출물 rubric으로 연결된다.
- [ ] 다음 lecture에서 재사용할 artifact를 지정했다.
