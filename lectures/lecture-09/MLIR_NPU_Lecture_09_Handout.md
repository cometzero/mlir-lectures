# MLIR NPU Lecture 09 - Pass Infrastructure

## 주제
Pass Infrastructure를 이해하고, AI/NPU compiler용 lowering pass pipeline 초안을 작성한다.

## 핵심 개념

- **Pass**: IR을 분석하거나 변환하는 compiler action unit.
- **OperationPass**: MLIR에서 operation을 current operation으로 삼아 실행되는 pass.
- **PassManager / OpPassManager**: operation nesting level별로 pass를 schedule하는 pipeline manager.
- **Dependent Dialects**: pass가 새 operation/type/attribute를 만들 수 있는 dialect를 사전에 등록하는 contract.
- **Analysis**: IR fact를 lazy/cached 방식으로 계산하는 helper class. IR mutation 후 preservation 여부를 명확히 해야 한다.
- **Instrumentation**: timing, statistics, IR printing, crash reproducer 등 pass pipeline debug 장치.

## NPU Compiler 관점

```text
StableHLO/TOSA
  -> Tensor/Linalg cleanup
  -> npu_graph legalization/fusion/layout/quant
  -> npu_kernel tiling/vectorization/memory-space assignment
  -> memref/bufferization
  -> DMA/barrier scheduling
  -> npu_rt descriptor/command-buffer emission
```

## 좋은 Pass 설계 체크리스트

| 항목 | 확인 질문 |
|---|---|
| Anchor | 이 pass는 module/function/kernel 중 어디에서 실행되는가? |
| Input dialect | 어떤 dialect를 읽는가? |
| Output dialect | 어떤 dialect를 생성하는가? |
| Dependent dialects | 새로 만드는 operation/type/attribute의 dialect를 등록했는가? |
| Analysis | shape/layout/liveness/dominance 중 무엇을 query하는가? |
| Preservation | IR 변경 후 어떤 analysis가 여전히 유효한가? |
| Failure | target invariant 위반 시 diagnostic과 signalPassFailure가 있는가? |
| Test | FileCheck positive/negative test가 있는가? |

## 실습 산출물

1. `builtin.module(func.func(...))` 형태의 pass pipeline annotation.
2. `npu-fuse-matmul-epilogue` pass skeleton의 TODO 작성.
3. MatMul+Bias+ReLU를 대상으로 NPU lowering pass pipeline 초안 작성.
4. 각 pass별 legality check와 regression test 전략 작성.

## 참고 명령

```bash
mlir-opt 01_pass_pipeline_input.mlir \
  --pass-pipeline='builtin.module(func.func(cse,canonicalize))'

mlir-opt 01_pass_pipeline_input.mlir \
  --pass-pipeline='builtin.module(func.func(cse,canonicalize))' \
  --mlir-print-ir-after-change --mlir-timing --mlir-pass-statistics
```

## References

- MLIR Pass Infrastructure: https://mlir.llvm.org/docs/PassManagement/
- MLIR Using mlir-opt: https://mlir.llvm.org/docs/Tutorials/MlirOpt/
- MLIR Passes: https://mlir.llvm.org/docs/Passes/
- MLIR Language Reference: https://mlir.llvm.org/docs/LangRef/
