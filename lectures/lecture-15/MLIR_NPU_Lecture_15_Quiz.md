# MLIR NPU Compiler - Lecture 15 Quiz

1. Transform Dialect에서 payload IR과 transform IR의 차이를 설명하세요.
2. `tile_using_for` 이후 원래 handle을 계속 쓰면 왜 위험한가요?
3. NPU native tile이 16x16x32일 때 `vector.contract`의 lhs/rhs/acc vector shape를 제안하세요.
4. Bias+ReLU epilogue fusion이 DRAM traffic을 줄이는 이유를 설명하세요.
5. `silenceable failure`와 `definite failure`의 차이를 NPU fallback 관점에서 설명하세요.
6. Transform Dialect가 pass infrastructure를 대체하지 않는 이유를 설명하세요.
7. SRAM footprint 계산 시 double buffering을 포함해야 하는 이유는 무엇인가요?
8. `-transform-dialect-check-uses`가 잡아낼 수 있는 문제를 예로 드세요.
9. unsupported K remainder에 대한 fallback 세 가지를 제안하세요.
10. `npu.schedule.tensorize_to_npu` custom transform op를 설계한다면 필요한 attribute를 쓰세요.
