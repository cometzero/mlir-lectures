# MLIR NPU Compiler - Lecture 08 Quiz

## Questions
1. `scf.for`의 lower bound, upper bound, step, induction variable은 각각 어떤 역할인가?
2. `scf.for`의 `iter_args`와 `scf.yield`는 어떤 관계인가?
3. Affine dialect가 SCF보다 loop analysis에 유리한 이유는 무엇인가?
4. MatMul에서 `m0/n0/k0` tile loop의 일반적인 순서와 각 loop의 data reuse를 설명하라.
5. `TM=32`, `TN=64`, `TK=64`, int8 input, int32 accumulator일 때 A/B/Accumulator tile 크기를 계산하라.
6. Double buffering에서 wait/barrier를 너무 빨리 넣으면 어떤 문제가 생기는가?
7. Edge tile에서 compiler가 보장해야 하는 correctness 조건은 무엇인가?
8. DMA load와 matmul compute 사이 hazard는 무엇이며 어떻게 표현할 수 있는가?
9. SCF로 표현하기 좋은 부분과 Affine으로 표현하기 좋은 부분을 NPU compiler 예시로 구분하라.
10. `linalg.matmul -> SCF/Affine -> npu dialect` regression test가 확인할 최소 항목 3가지는?

## Answer sketch
1. Loop iteration domain and stride; IV is the per-iteration SSA index value.
2. `iter_args` initializes loop-carried values; `scf.yield` returns the next values.
3. Affine restrictions make bounds/accesses analyzable for dependence and loop transforms.
4. Usually `m0 -> n0 -> k0`; C tile is reused across K, A/B stream per K.
5. A=2048 B, B=4096 B, Acc=8192 B, double-buffered total ~= 20480 B.
6. It serializes DMA and compute, reducing overlap.
7. Valid read/write range, correct accumulation, and no out-of-bounds access.
8. Consume-before-DMA-complete or overwrite-before-consume; use wait/barrier/token dependency.
9. SCF for dynamic runtime schedule; Affine for static tiled loops and affine memory indexing.
10. Loop order, tile sizes, accumulator lifetime, DMA/barrier ordering, edge policy, types.
