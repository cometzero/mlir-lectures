# MLIR_NPU_Lecture_05_Materials

Lecture 05 of the AI/NPU Compiler MLIR course.

## Topic
Dialect System - splitting frontend, graph, kernel and runtime dialects for NPU compiler design.

## Files
- `MLIR_NPU_Lecture_05_Slides.pptx` / `.pdf`: lecture slides
- `MLIR_NPU_Lecture_05_Notes.docx` / `.pdf`: detailed lecture notes
- `MLIR_NPU_Lecture_05_Worksheet.docx` / `.pdf`: exercises
- `MLIR_NPU_Lecture_05_Handout.md`: student handout
- `MLIR_NPU_Lecture_05_Quiz.md`: quiz and answer key
- `examples/MLIR_NPU_Lecture_05_Dialect_Examples.mlir`: mixed dialect and pseudo NPU dialect examples
- `examples/MLIR_NPU_Lecture_05_NPU_Dialect_TD.td`: ODS/TableGen pseudo skeleton
- `examples/MLIR_NPU_Lecture_05_Pass_Dependent_Dialects.cpp`: pass dependent dialect pseudo skeleton
- `assets/*.png`: diagrams

## Lab output
1. Dialect layer classification table
2. `npu_kernel.matmul` operation schema
3. ODS pseudo definition and verifier checklist
4. Frontend -> graph -> kernel -> runtime lowering boundary matrix
