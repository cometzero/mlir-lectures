# MLIR NPU Compiler 20강 Syllabus

이 문서는 전체 커리큘럼의 진행 순서와 lecture별 실습 산출물을 요약합니다.

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

## 단계별 학습 흐름

### Part 1. MLIR 기본기와 도구 (Lecture 01~04)
- MLIR/NPU compiler 전체 그림
- Operation/SSA/type/attribute/region/block 구조
- `mlir-opt`, `mlir-translate`, FileCheck 기반 regression loop

### Part 2. Dialect와 입력 IR (Lecture 05~08)
- Dialect 설계와 abstraction boundary
- StableHLO/TOSA/Linalg/SCF/Affine 기반 모델 lowering
- tile loop, DMA, command region으로 이어지는 사고방식

### Part 3. Pass, Pattern, Conversion, Bufferization (Lecture 09~13)
- Pass pipeline과 rewrite pattern
- Dialect conversion과 legal op set
- Tensor에서 memref로 넘어가는 bufferization/lifetime/memory space

### Part 4. NPU 최적화와 스케줄링 (Lecture 14~17)
- Vector dialect, tensorization, Transform dialect
- Quantization, fusion, layout propagation
- 비용 모델과 target legality 중심의 최적화 판단

### Part 5. Custom Dialect, Runtime, Capstone (Lecture 18~20)
- `npu.matmul`, `npu.dma`, `npu.barrier` ODS 설계
- runtime ABI/codegen/command buffer
- mini NPU compiler design spec 발표

## 각 강의 공통 산출물

- Annotated IR: 예제 IR에 operand/type/attribute/pass boundary 주석 추가
- Legality checklist: target에서 허용/불허할 조건 정리
- Regression sketch: positive/negative FileCheck 또는 verifier test 초안
- Design note: 다음 lowering stage로 넘길 contract 명시
