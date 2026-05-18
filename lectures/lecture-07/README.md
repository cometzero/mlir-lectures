# MLIR NPU Lecture 07 Materials

Topic: Linalg-on-Tensors

Core output: express MatMul and Conv2D using `linalg.generic`, then analyze them from an NPU compiler perspective.

## Files

- MLIR_NPU_Lecture_07_Slides.pptx / .pdf
- MLIR_NPU_Lecture_07_Notes.docx / .pdf
- MLIR_NPU_Lecture_07_Worksheet.docx / .pdf
- MLIR_NPU_Lecture_07_Handout.md
- MLIR_NPU_Lecture_07_Quiz.md
- examples/*.mlir
- assets/*.png

## Suggested lesson flow

1. Review Linalg-on-Tensors pipeline position.
2. Decompose `linalg.generic` into maps, iterators, ins/outs, and region.
3. Walk through MatMul generic.
4. Walk through Conv2D NHWC/HWCF generic.
5. Complete worksheet tile footprint and NPU mapping exercises.

## References

- https://mlir.llvm.org/docs/Dialects/Linalg/
- https://mlir.llvm.org/docs/Tutorials/transform/Ch0/
- https://mlir.llvm.org/docs/Dialects/TensorOps/
