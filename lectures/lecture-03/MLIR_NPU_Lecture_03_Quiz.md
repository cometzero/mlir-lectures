# MLIR NPU Compiler - Lecture 03 Quiz

## 문제

1. MLIR의 기본 중첩 구조를 `Operation`, `Region`, `Block` 순서로 설명하세요.
2. Region의 의미를 최종적으로 결정하는 것은 region 자체인가요, parent operation인가요?
3. MLIR에서 LLVM PHI node에 해당하는 merge value는 보통 무엇으로 표현되나요?
4. `scf.for`의 induction variable은 어떤 형태로 loop body에 전달되나요?
5. SSACFG region의 block 마지막에는 무엇이 필요하나요?
6. NPU command region에서 `npu.dma.async`가 token을 반환하는 이유는 무엇인가요?
7. `tile_m = 128` 같은 고정 tile 크기는 operand와 attribute 중 어느 쪽이 더 적합한가요?
8. region 내부에서 정의된 SRAM view가 region 밖에서 임의로 사용되면 어떤 문제가 생길 수 있나요?

## 정답 예시

1. Operation은 0개 이상의 Region을 가질 수 있고, Region은 Block list이며, Block은 argument와 operation list를 가진다.
2. Parent operation이 region의 semantics, argument, terminator, result 관계를 정의한다.
3. Block argument. Predecessor branch operand가 successor block argument로 전달된다.
4. Loop body region의 block argument.
5. Terminator operation. 예: `func.return`, `cf.br`, `scf.yield`, custom `npu.yield`.
6. DMA completion과 compute scheduling 사이의 dependency를 IR에서 명시하기 위해서.
7. Attribute. 컴파일 타임 scheduling/configuration metadata이기 때문이다. 단, runtime에 바뀐다면 operand가 될 수 있다.
8. dominance/scope 위반, SRAM lifetime 오류, resource hazard, command-buffer serialization 오류가 발생할 수 있다.
