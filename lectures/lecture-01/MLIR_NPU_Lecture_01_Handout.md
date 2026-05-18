# MLIR for AI/NPU Compiler - Lecture 01 Handout

## Topic
MLIR와 AI/NPU Compiler 큰 그림

## Core Pipeline

```text
Model Frontend -> StableHLO/TOSA -> Tensor/Linalg -> Bufferization/MemRef -> SCF/Affine/Vector -> Custom NPU Dialect -> Runtime ABI/Command Buffer
```

## Lab Output
Create a one-page NPU compiler lowering pipeline for MatMul+Bias+ReLU.

| Stage | Input IR | Output IR | Decision | Check |
|---|---|---|---|---|
| Import | | | | |
| Legalize | | | | |
| Canonicalize | | | | |
| Fuse | | | | |
| Tile | | | | |
| Bufferize | | | | |
| Schedule | | | | |
| Emit | | | | |

## References
- MLIR Language Reference: https://mlir.llvm.org/docs/LangRef/
- MLIR Understanding the IR Structure: https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/
- MLIR Using mlir-opt: https://mlir.llvm.org/docs/Tutorials/MlirOpt/
- MLIR Dialects: https://mlir.llvm.org/docs/Dialects/
- MLIR Bufferization: https://mlir.llvm.org/docs/Bufferization/
- StableHLO Specification: https://openxla.org/stablehlo/spec
- TOSA: https://www.mlplatform.org/tosa/
- MLIR TOSA Dialect: https://mlir.llvm.org/docs/Dialects/TOSA/
