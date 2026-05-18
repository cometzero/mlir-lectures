# MLIR NPU Compiler - Lecture 04 Handout

## Topic
mlir-opt, mlir-translate, FileCheck: pass pipeline execution and regression testing.

## Mental model

```text
input.mlir
  -> mlir-opt --pass-pipeline='builtin.module(func.func(cse,canonicalize))'
  -> transformed.mlir
  -> FileCheck input.mlir
  -> regression pass/fail
```

## Essential commands

```bash
mlir-opt input.mlir
mlir-opt input.mlir --pass-pipeline='builtin.module(func.func(cse,canonicalize))'
mlir-opt input.mlir --mlir-print-ir-after-all
mlir-opt input.mlir --mlir-print-ir-after-change
mlir-opt input.mlir -split-input-file -verify-diagnostics
mlir-translate --mlir-to-llvmir llvm_dialect.mlir
```

## FileCheck core directives

| Directive | Use case |
|---|---|
| CHECK-LABEL | start of an independent function/test block |
| CHECK | ordered match |
| CHECK-NEXT | exact next line match |
| CHECK-SAME | same line match, useful for attributes/types |
| CHECK-NOT | absence between positive checks |
| CHECK-DAG | unordered set, useful for scheduling-independent output |
| %[[V:.*]] | capture SSA name without depending on printed numbering |

## NPU compiler regression checklist

- Does the pass remove illegal high-level ops after lowering?
- Does the pass preserve shape, element type, quantization scale, and layout metadata?
- Does DMA appear before compute, and barrier/await before consuming SRAM data?
- Does fusion preserve numerical order, especially requantization and activation placement?
- Are negative cases named with a `negative_` prefix and checked by diagnostics?

## Exercise output
Prepare a FileCheck test for pseudo `npu-fuse-matmul-bias-relu` and explain which invariants your check lines verify.
