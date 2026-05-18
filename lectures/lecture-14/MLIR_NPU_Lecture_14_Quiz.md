# Lecture 14 Quiz - Vector Dialect and Tensorization

## Multiple Choice

1. `vector.contract`를 NPU matmul로 tensorization하기 전에 가장 먼저 확인해야 할 것은?
   - A. `func.func` 이름
   - B. indexing maps, iterator types, vector shape, dtype
   - C. source file path
   - D. debug flag

2. `vector.transfer_read`의 `permutation_map`이 중요한 이유는?
   - A. pass 이름을 줄이기 위해
   - B. memory dimension과 vector dimension의 mapping을 정의하기 때문
   - C. tensor rank를 항상 1로 만들기 때문
   - D. accumulator type을 바꾸기 때문

3. native NPU가 M16N16K32 int8 matmul만 지원할 때 `vector<16x48xi8> x vector<48x16xi8>`은 보통 어떻게 처리하는가?
   - A. 그대로 native op 하나로 emit
   - B. K dimension split/unroll 후 여러 native op로 lower하거나 reject
   - C. 항상 CPU fallback
   - D. dtype만 i32로 바꾸면 됨

## Short Answer

4. vectorization과 tensorization의 차이를 한 문단으로 설명하세요.

5. mask edge tile을 지원하지 않는 NPU에서 사용할 수 있는 두 가지 처리 방법을 쓰세요.

6. `vector.contract`의 accumulator type이 NPU correctness에 중요한 이유를 설명하세요.

## Coding Prompt

아래 vector.contract를 `npu_kernel.matmul` pseudo op로 lowering하세요.

```mlir
%c = vector.contract #trait %a, %b, %acc
  : vector<16x32xi8>, vector<32x16xi8> into vector<16x16xi32>
```
