# MLIR NPU Compiler - Lecture 03 Handout

## 주제
Region과 Block: NPU kernel/command region pseudo IR 설계

## 1. 핵심 개념

MLIR IR은 `Operation -> Region -> Block -> Operation` 형태로 재귀적으로 중첩됩니다. 2강에서 operation의 operand/result/type/attribute를 읽었다면, 3강에서는 operation이 소유하는 region과 region 내부 block의 제어흐름을 읽습니다.

```text
Operation
  Region[]
    Block[]
      BlockArgument[]
      Operation[]
      Terminator
```

## 2. Region

Region은 parent operation 아래에 중첩된 block list입니다. 함수 body, loop body, graph body, NPU kernel body, command-buffer body를 표현할 수 있습니다. 중요한 점은 region 자체가 의미를 결정하지 않고, parent operation의 semantics가 region argument, terminator, result의 의미를 정의한다는 것입니다.

## 3. Block

Block은 block argument와 operation sequence를 가집니다. SSACFG region에서는 마지막 operation이 terminator여야 하며, terminator가 successor block 또는 parent operation으로 control flow를 전달합니다. MLIR은 LLVM PHI node 대신 block argument로 merge value를 표현합니다.

## 4. NPU Compiler 관점

NPU compiler에서 region/block은 다음 설계에 직접 연결됩니다.

- Kernel body: tiled matmul, convolution, attention kernel의 semantic unit
- Command region: DMA load, barrier, compute, store command sequence
- Async dependency: DMA token과 barrier ordering
- Memory scope: DRAM/SRAM/accumulator view의 lifetime과 visibility
- Verification: terminator, block argument type, dominance, resource hazard 검사

## 5. 실습 산출물

MatMul tile 하나에 대해 다음 sequence를 pseudo IR region으로 작성합니다.

```text
1. DMA A tile: DRAM -> SRAM_A
2. DMA B tile: DRAM -> SRAM_B
3. await/barrier
4. NPU matmul: SRAM_A x SRAM_B -> ACC
5. requant/relu: ACC -> SRAM_C
6. DMA C tile: SRAM_C -> DRAM
7. yield/return terminator
```

## 6. 체크리스트

- 모든 block에 terminator가 있는가?
- region argument는 parent operation semantics와 일치하는가?
- block argument는 predecessor branch operand와 type/count가 맞는가?
- region 내부 value가 region 밖으로 불법 escape하지 않는가?
- async DMA token이 compute 전에 await되는가?
- compile-time property는 attribute, runtime value는 operand로 표현했는가?

## 참고자료

- MLIR Language Reference: https://mlir.llvm.org/docs/LangRef/
- Understanding the IR Structure: https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/
- MLIR Rationale - Block Arguments vs PHI nodes: https://mlir.llvm.org/docs/Rationale/Rationale/
- SCF Dialect: https://mlir.llvm.org/docs/Dialects/SCFDialect/
