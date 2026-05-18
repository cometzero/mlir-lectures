# MLIR NPU Compiler - Lecture 18 Handout

## Topic
Custom NPU Dialect: `npu.matmul`, `npu.dma`, `npu.barrier` ODS design.

## 핵심 메시지
Custom dialect는 단순 문법 확장이 아니라 NPU target contract입니다. tile shape, layout, SRAM/ACCUM memory space, DMA direction, token/barrier dependency를 operation/type/attribute/verifier로 명확히 표현해야 합니다.

## 권장 abstraction split
| Layer | Responsibility | Example |
|---|---|---|
| `npu_graph` | fusion/layout/quant semantic contract | `npu_graph.matmul_epilogue` |
| `npu_kernel` or `npu` | tile/DMA/barrier/accumulator contract | `npu.matmul`, `npu.dma`, `npu.barrier` |
| `npu_rt` | command buffer and ABI | `npu_rt.launch` |

## 18강 실습 산출물
1. `npu.matmul` ODS skeleton
2. `npu.dma` ODS skeleton
3. `npu.barrier` ODS skeleton
4. Verifier checklist
5. Positive/negative FileCheck tests
