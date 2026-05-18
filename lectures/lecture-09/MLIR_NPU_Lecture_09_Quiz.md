# MLIR NPU Lecture 09 - Quiz

## 객관식/단답

1. MLIR PassManager가 pass를 operation nesting level별로 schedule하는 이유를 두 가지 쓰세요.
2. `OperationPass<func::FuncOp>`와 `OperationPass<>`의 차이를 설명하세요.
3. Pass가 `npu_graph.matmul` op를 생성한다면 어떤 메서드에서 어떤 정보를 등록해야 하나요?
4. `signalPassFailure()`가 호출되면 pipeline은 어떻게 동작하나요?
5. `--mlir-print-ir-after-change`가 useful한 상황을 설명하세요.
6. Analysis preservation을 잘못 선언했을 때 발생할 수 있는 문제를 NPU compiler 관점으로 설명하세요.
7. 다음 pipeline에서 `cse`와 `canonicalize`가 실행되는 anchor를 표시하세요.

```text
builtin.module(func.func(cse,canonicalize),npu-infer-layout)
```

## 설계 문제

MatMul + Bias + ReLU를 NPU target으로 낮추기 위한 pass pipeline을 8개 이상 작성하세요.
각 pass마다 다음 항목을 기입하세요.

| Pass | Anchor | Input dialect | Output dialect | Required analysis | Preserved analysis | Test |
|---|---|---|---|---|---|---|

## 정답 가이드

- dependent dialect는 pass가 새로 생성할 dialect entity를 사전에 알려 multi-threaded pipeline 실행 전에 dialect가 load되도록 하는 contract이다.
- `signalPassFailure()`는 graceful failure mechanism이며, 이후 pipeline pass 실행은 중단되고 top-level `PassManager::run`은 failure를 반환한다.
- NPU pipeline은 graph-level fusion/layout을 bufferization보다 앞에 배치하는 것이 일반적으로 유리하다.
