# MLIR NPU Lecture 07 - Quiz

## 1. 개념 확인

1. `linalg.generic`의 `indexing_maps`는 무엇을 표현하는가?
2. MatMul에서 `k` dimension이 `reduction` iterator인 이유는 무엇인가?
3. Tensor value semantic을 bufferization 전까지 유지하는 장점은 무엇인가?
4. `linalg.matmul` named op와 `linalg.generic` 표현의 차이를 설명하라.
5. Conv2D NHWC/HWCF에서 output channel `f`는 parallel인가 reduction인가? 이유는?

## 2. 짧은 답안

6. C[M,N] = sum_k A[M,K] * B[K,N]에 대한 세 개의 indexing map을 작성하라.
7. Conv2D input map에서 stride=2라면 `oh + kh`는 어떻게 바뀌는가?
8. Linalg 단계에서 epilogue fusion 여부를 판단해야 하는 이유를 NPU DRAM traffic 관점에서 설명하라.

## 3. 실습형 문제

9. 다음 MatMul tile에 대해 A/B/C tile element 수를 계산하라: M=32, N=64, K=128, FP16.
10. Output tile OH=8, OW=8, F=16, KH=3, KW=3, C=16인 Conv에 대해 input halo tile과 filter tile 크기를 계산하라.

## 정답 키워드

- indexing_maps: loop IV tuple to operand indices.
- MatMul: (m,n,k) -> A(m,k), B(k,n), C(m,n); iterator_types = parallel, parallel, reduction.
- Conv: output axes parallel; kh/kw/c reduction.
- Linalg-on-Tensors: tile/fuse/vectorize decisions before physical buffer layout.
