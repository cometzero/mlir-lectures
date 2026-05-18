# MLIR NPU Compiler Study Lectures

GPT-5.5로 생성한 MLIR/NPU Compiler 20강 스터디 자료를 강의와 실습이 가능한 형태로 정리한 저장소입니다.

이 저장소는 단순 자료 보관이 아니라, 강의자가 바로 따라가며 수업을 진행할 수 있도록 다음 구조를 제공합니다.

- `lectures/lecture-XX/`: 각 장의 원본 슬라이드, 노트, 워크시트, 핸드아웃, 퀴즈, 예제, 자산
- `lectures/lecture-XX/LESSON.md`: 90~120분 강의 진행안, 실습 흐름, review 질문
- `docs/syllabus.md`: 20강 전체 커리큘럼
- `docs/teaching-guide.md`: 강의 운영 방식과 평가/피드백 가이드
- `docs/lab-environment.md`: MLIR 실습 환경 준비와 실행 방법
- `scripts/run_lesson.sh`: lecture별 stock-safe 예제 실행/점검 helper
- `scripts/run_all_lessons.sh`: 20개 lecture smoke test helper
- `scripts/verify_materials.py`: 자료 누락 여부 검증 helper
- `course-package/`: 전체 코스 요약 패키지

## 빠른 시작

```bash
git clone https://github.com/cometzero/mlir-lectures.git
cd mlir-lectures
python3 scripts/verify_materials.py
```

MLIR 도구가 설치되어 있다면:

```bash
mlir-opt --version
FileCheck --version
scripts/run_lesson.sh 04 --try-mlir
scripts/run_all_lessons.sh
```

MLIR 도구가 아직 없다면 먼저 `docs/lab-environment.md`를 확인하세요. 많은 예제는 custom/pseudo NPU dialect를 포함하므로, 모든 `.mlir` 파일이 vanilla `mlir-opt`에서 바로 성공하는 것을 목표로 하지 않습니다. `scripts/run_lesson.sh --try-mlir`는 stock MLIR toolchain에서 검증 가능한 예제만 실행하고, custom backend가 필요한 예제는 `SKIP`으로 표시합니다. 강의에서는 SKIP된 예제를 통해 필요한 dialect registration, verifier, lowering boundary를 토론할 수 있습니다.

## 커리큘럼

| Lecture | Topic | Materials | Lab |
|---:|---|---|---|
| 01 | MLIR and AI/NPU Compiler Big Picture | [`lectures/lecture-01`](lectures/lecture-01/) | 2 example files |
| 02 | MLIR IR Basic Structure | [`lectures/lecture-02`](lectures/lecture-02/) | 1 example files |
| 03 | Region and Block: NPU kernel/command region pseudo IR | [`lectures/lecture-03`](lectures/lecture-03/) | 2 example files |
| 04 | mlir-opt, mlir-translate, FileCheck | [`lectures/lecture-04`](lectures/lecture-04/) | 9 example files |
| 05 | Dialect System | [`lectures/lecture-05`](lectures/lecture-05/) | 3 example files |
| 06 | StableHLO, TOSA, and AI Model Input IR | [`lectures/lecture-06`](lectures/lecture-06/) | 4 example files |
| 07 | Linalg-on-Tensors | [`lectures/lecture-07`](lectures/lecture-07/) | 7 example files |
| 08 | SCF/Affine Loop | [`lectures/lecture-08`](lectures/lecture-08/) | 8 example files |
| 09 | Pass Infrastructure | [`lectures/lecture-09`](lectures/lecture-09/) | 9 example files |
| 10 | Rewrite Pattern | [`lectures/lecture-10`](lectures/lecture-10/) | 16 example files |
| 11 | Dialect Conversion - NPU legal op set and unsupported op lowering | [`lectures/lecture-11`](lectures/lecture-11/) | 11 example files |
| 12 | Lecture 12 - Bufferization: Tensor to MemRef, Alias, and Lifetime Analysis | [`lectures/lecture-12`](lectures/lecture-12/) | 10 example files |
| 13 | MemRef, Layout, Memory Space | [`lectures/lecture-13`](lectures/lecture-13/) | 8 example files |
| 14 | Vector Dialect and Tensorization | [`lectures/lecture-14`](lectures/lecture-14/) | 10 example files |
| 15 | Transform Dialect | [`lectures/lecture-15`](lectures/lecture-15/) | 9 example files |
| 16 | Quantization for AI/NPU Compiler | [`lectures/lecture-16`](lectures/lecture-16/) | 9 example files |
| 17 | Fusion, Layout Propagation, Transformer MLP fusion candidate analysis | [`lectures/lecture-17`](lectures/lecture-17/) | 10 example files |
| 18 | Custom NPU Dialect: npu.matmul / npu.dma / npu.barrier ODS Design | [`lectures/lecture-18`](lectures/lecture-18/) | 11 example files |
| 19 | Runtime / ABI / Codegen | [`lectures/lecture-19`](lectures/lecture-19/) | 11 example files |
| 20 | Lecture 20 - Capstone: Mini NPU Compiler Design Spec | [`lectures/lecture-20`](lectures/lecture-20/) | 11 example files |

## 추천 진행 방식

1. `docs/syllabus.md`로 전체 흐름을 먼저 파악합니다.
2. 각 lecture에서 `LESSON.md` -> `Handout` -> `Slides` -> `examples/` -> `Worksheet` -> `Quiz` 순서로 진행합니다.
3. 실습은 항상 "IR을 읽는다 -> target constraint를 찾는다 -> 변환/검증 기준을 적는다 -> regression test 또는 checklist로 고정한다" 순서로 진행합니다.
4. 마지막 20강에서는 앞선 산출물을 모아 mini NPU compiler design spec을 완성합니다.

## 저장소 구성 원칙

- 원본 자료(PDF/PPTX/DOCX/MD/예제/이미지)를 보존합니다.
- macOS metadata(`__MACOSX`, `.DS_Store`, `._*`)와 중복 zip archive는 제외했습니다.
- GitHub 저장소에서 바로 읽기 쉬운 Markdown guide를 추가했습니다.
