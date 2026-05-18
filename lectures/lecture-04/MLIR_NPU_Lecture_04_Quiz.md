# MLIR NPU Compiler - Lecture 04 Quiz

1. `mlir-opt`를 아무 pass 없이 실행했을 때 얻는 가장 기본적인 효과는 무엇인가요?
2. `--pass-pipeline='builtin.module(func.func(cse,canonicalize))'`에서 `builtin.module`과 `func.func`의 의미는 무엇인가요?
3. NPU compiler pass test에서 `CHECK-NOT: stablehlo.dot_general`이 유용한 이유는 무엇인가요?
4. `CHECK-DAG`를 써야 하는 경우와 쓰면 위험한 경우를 각각 설명하세요.
5. `-verify-diagnostics` 기반 negative test가 필요한 NPU compiler 사례를 2개 드세요.
6. `mlir-translate`와 `mlir-opt`의 책임을 구분해서 설명하세요.

## Suggested answers

1. 입력 textual/bytecode MLIR을 parse하고 verifier를 실행한 뒤 textual IR로 serialize하여 well-formed 여부를 확인한다.
2. pass manager가 어떤 operation 단위에 pass를 anchor할지 지정한다. module pass와 function nested pass를 구분한다.
3. lowering 후 남아 있으면 안 되는 frontend op가 제거되었는지 확인할 수 있다.
4. 순서가 의미 없는 독립 op는 DAG가 적합하다. order가 correctness인 DMA/barrier/compute sequence는 일반 CHECK 순서가 더 안전하다.
5. unsupported op, invalid layout, illegal quantization scale, tile size constraint violation, missing memory space 등.
6. mlir-opt는 MLIR 내부 pass 실행/디버깅 도구이고, mlir-translate는 MLIR과 LLVM IR/SPIR-V 같은 외부 표현 사이 translation 도구다.
