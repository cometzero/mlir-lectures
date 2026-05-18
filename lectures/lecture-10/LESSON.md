# Lecture 10 - Rewrite Pattern

이 파일은 원본 자료를 강의자가 바로 진행할 수 있도록 재구성한 진행안입니다. 원본 PDF/PPTX/DOCX는 보존되어 있고, 실습은 `examples/`와 `Worksheet`를 기준으로 진행합니다.

## 1. 학습 목표

- 이번 장의 핵심 주제인 `Rewrite Pattern`를 MLIR/NPU compiler pipeline 안에서 설명할 수 있습니다.
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

- 슬라이드: `MLIR_NPU_Lecture_10_Slides.pdf`
- 강의 노트: `MLIR_NPU_Lecture_10_Notes.pdf`
- 핸드아웃: `MLIR_NPU_Lecture_10_Handout.md`
- 워크시트: `MLIR_NPU_Lecture_10_Worksheet.pdf`
- 퀴즈: `MLIR_NPU_Lecture_10_Quiz.md`

## 4. 강의 포인트

- 주제
- 학습 목표
- 핵심 개념
- 추천 실습 순서
- 한 줄 요약

## 5. 실습 파일

- `examples/01_linalg_matmul_add_relu_before.mlir` - MLIR IR 예제
- `examples/01_matmul_add_relu_before.mlir` - MLIR IR 예제
- `examples/02_npu_fused_matmul_epilogue_after.mlir` - MLIR IR 예제
- `examples/02_npu_graph_fused_after.mlir` - MLIR IR 예제
- `examples/03_cpp_rewrite_pattern_skeleton.cpp` - C++/ODS skeleton
- `examples/03_matmul_add_relu_fusion_pattern.cpp` - C++/ODS skeleton
- `examples/04_pdl_matmul_add_relu_pattern.pdll`
- `examples/04_rewrite_pass_skeleton.cpp` - C++/ODS skeleton
- `examples/05_drr_pattern_skeleton.td` - C++/ODS skeleton
- `examples/05_pdl_matmul_add_relu_pattern.mlir` - MLIR IR 예제
- `examples/06_filecheck_fusion_test.mlir` - MLIR IR 예제
- `examples/07_greedy_driver_pass.cpp` - C++/ODS skeleton
- `examples/07_negative_no_fusion_test.mlir` - MLIR IR 예제
- `examples/08_fusion_contract_checklist.md` - 체크리스트/설명 문서
- `examples/08_negative_no_fusion.mlir` - MLIR IR 예제
- `examples/run_lesson10.sh` - 실습 실행 스크립트

## 6. 실행 방법

공통 환경은 루트의 `docs/lab-environment.md`를 먼저 확인하세요.

```bash
cd lectures/lecture-10
bash examples/run_lesson10.sh
```


주의: 일부 파일은 교육용 pseudo dialect 또는 skeleton 코드입니다. 실제 MLIR build에 custom dialect가 등록되어 있지 않으면 parse/verify가 실패할 수 있습니다. 실패 자체를 오류로만 보지 말고, 어떤 dialect/type/op registration이 필요한지 토론 자료로 사용하세요.

## 7. 수강자 산출물

- 핵심 IR 또는 설계 요소에 주석을 단 annotated example
- worksheet 답안 또는 checklist
- positive/negative regression test 초안
- NPU target legality, layout, memory, quantization 또는 runtime ABI 관점의 리뷰 메모

## 8. Review 질문

- `PatternRewriter`를 사용하는 가장 중요한 이유는?
- A. code generation 속도를 높이기 위해
- B. pattern driver가 IR mutation 상태를 추적할 수 있게 하기 위해
- C. 모든 op를 LLVM IR로 바로 변환하기 위해
- D. FileCheck를 자동 생성하기 위해

## 9. 강의자 체크리스트

- [ ] 수강자가 `operand / attribute / type / region / pass boundary`를 구분해서 설명할 수 있다.
- [ ] 실습 산출물이 NPU target constraint를 최소 1개 이상 명시한다.
- [ ] 예제의 실패/성공 기준이 FileCheck, verifier, checklist 또는 산출물 rubric으로 연결된다.
- [ ] 다음 lecture에서 재사용할 artifact를 지정했다.
