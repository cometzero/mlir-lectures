# MLIR / NPU Compiler Course - Lecture 05 Quiz

## Questions

1. MLIR Dialect가 정의할 수 있는 대표 구성 요소 3가지를 쓰세요.
2. `npu_graph` dialect와 `npu_kernel` dialect를 분리하는 이유를 쓰세요.
3. `tile_shape=[16,16,32]`는 operand, attribute, type 중 어디에 두는 것이 자연스럽습니까? 이유는 무엇입니까?
4. `npu_rt.launch`가 frontend dialect에 있으면 왜 좋지 않습니까?
5. ODS/TableGen을 사용하는 이유를 2가지 쓰세요.
6. verifier에서 검사해야 할 NPU target legality rule을 4개 쓰세요.
7. pass가 특정 dialect의 op를 새로 생성할 때 dialect registration과 dependent dialect 선언이 중요한 이유를 설명하세요.
8. Dialect Conversion에서 ConversionTarget, RewritePattern, TypeConverter의 역할을 간단히 설명하세요.
9. `npu_kernel.dma_load`가 MemoryEffectOpInterface를 구현하면 pass 입장에서 어떤 장점이 있습니까?
10. custom dialect를 너무 빨리 도입했을 때 잃을 수 있는 MLIR 공통 최적화 기회를 쓰세요.

## Answer Key

1. Operations, Types, Attributes. 추가로 traits, interfaces, verifiers, parser/printer도 포함할 수 있습니다.
2. graph dialect는 fusion/layout/quant 등 operator-level legality를 다루고, kernel dialect는 tile/DMA/barrier/SRAM/accumulator 같은 hardware scheduling을 다룹니다. 두 단계의 verifier와 lowering target이 다릅니다.
3. 보통 attribute입니다. tile_shape는 compile-time scheduling decision이고 dataflow value가 아니기 때문입니다. 단 dynamic tiling이면 operand가 될 수 있습니다.
4. runtime launch는 ABI/descriptor/stream/sync와 연결되는 낮은 수준의 개념이므로 model semantic을 보존해야 하는 frontend layer를 오염시킵니다.
5. declarative op definition으로 boilerplate를 줄이고, parser/printer/verifier/builder/doc generation/test 구조를 일관되게 만들 수 있습니다.
6. shape legality, dtype/accumulator legality, layout compatibility, SRAM capacity, DMA alignment, barrier dependency, command buffer ABI size 등.
7. MLIRContext는 dialect entity를 만들기 전에 해당 dialect를 알아야 하며, pass pipeline과 멀티스레드 실행에서 안전한 registration을 보장하기 위해 필요합니다.
8. ConversionTarget은 합법/불법 op를 정의하고, RewritePattern은 불법 op를 합법 op로 바꾸며, TypeConverter는 타입 변환 규칙을 제공합니다.
9. side-effect analysis, scheduling, CSE/DCE 제한, dependency analysis가 명확해져 DMA와 compute reorder의 안전성을 판단할 수 있습니다.
10. linalg/vector/scf/memref에 이미 존재하는 tiling, fusion, canonicalization, vectorization, bufferization 인프라를 재사용하기 어려워집니다.
