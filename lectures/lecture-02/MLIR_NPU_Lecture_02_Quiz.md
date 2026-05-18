# MLIR NPU Compiler - Lecture 02 Quiz

1. In MLIR, what is the difference between an operation result and a block argument?
2. Why should a runtime pointer address not usually be modeled as an attribute?
3. In `tensor<1x128x768xf16>`, which information is shape and which is element type?
4. What is the benefit of generic operation assembly when debugging a custom NPU dialect?
5. Name three NPU compiler decisions that can be represented as attributes.
6. Why is preserving `loc(...)` important after lowering?
7. What kind of type information is needed to distinguish DRAM and SRAM buffers?
8. When might `!npu.token` be useful?
9. Which value in the matmul example is a good allocation candidate after bufferization?
10. What verifier rule would you add to `npu.matmul` regarding accumulator type?

## Answer sketch

1. Operation result is produced by an op; block argument is introduced by a block/region interface.
2. It changes at runtime and belongs in an operand or runtime descriptor.
3. `1x128x768` is shape; `f16` is element type.
4. Generic assembly exposes operation name, operands, attributes, and type signature explicitly.
5. Tile size, memory space, layout, quant scale, iterator type.
6. It maps compiled IR errors and profiling hotspots back to model/source nodes.
7. MemRef memory space, custom type, or descriptor attribute depending on pipeline stage.
8. To model ordering/dependencies between DMA and compute commands.
9. `%init` from `tensor.empty` or a bufferized destination for `%filled/%out`.
10. For INT8 input, accumulator must be i32 or a target-approved accumulation type.
