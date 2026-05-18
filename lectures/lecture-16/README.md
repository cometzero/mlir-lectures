# MLIR NPU Lecture 16 Materials

Topic: Quantization for AI/NPU Compiler

This package contains Korean lecture notes, slides, worksheet, handout, quiz, quick reference, MLIR examples, Python calculators, and diagrams.

Core output:
- Understand `!quant.uniform`, `quant.qcast`, `quant.dcast`, `quant.scast`.
- Design INT8 MatMul quantization and requantization.
- Analyze accumulator width for NPU kernels.
- Define INT4 weight packing and group scale policy.
- Write verifier and FileCheck tests.

References:
- MLIR Quant Dialect: https://mlir.llvm.org/docs/Dialects/QuantDialect/
- MLIR Quantization Design: https://mlir.llvm.org/docs/Quantization/
- MLIR TOSA Dialect: https://mlir.llvm.org/docs/Dialects/TOSA/
- MLIR Passes: https://mlir.llvm.org/docs/Passes/
