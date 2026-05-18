# Lecture 18 Quiz

1. ODS가 custom dialect 설계에서 single source of truth가 되는 이유는?
2. `npu.matmul`에서 tile shape는 operand, attribute, type 중 무엇이 적합한가요?
3. token-based `npu.dma` 모델의 장단점은?
4. verifier가 reject해야 하는 조건과 cost model이 판단해야 하는 조건을 구분하세요.
5. `npu.barrier`에 scope attribute가 필요한 이유는?
6. `MemoryEffectsOpInterface`가 NPU command op에 중요한 이유는?
7. parser/printer round-trip test와 negative verifier test의 차이는?
8. `vector.contract -> npu.matmul` lowering legality 조건 4가지를 쓰세요.

## Suggested answers
1. op name, operands, results, attributes, traits, assembly format, verifier hook를 한 곳에 선언하고 generated accessor/build/verify code의 기반이 되기 때문입니다.
2. 정적 hardware fact라면 attribute가 적합합니다. value-dependent이면 operand가 필요합니다.
3. dependency analysis가 쉬우나, hardware tag/packet으로 lowering하는 별도 단계가 필요할 수 있습니다.
4. illegal shape/layout/memory space는 verifier, 느린 schedule은 cost model입니다.
5. DMA-only, compute-only, all-engine, cluster-level wait semantics를 구분하기 위해서입니다.
6. pass가 command를 reorder/canonicalize할 때 read/write effects를 보존해야 하기 때문입니다.
7. round-trip은 syntax stability, negative test는 diagnostics와 legality enforcement를 검증합니다.
8. tile shape, element types, layouts, accumulator type/width, memory spaces, mask/remainder policy.
