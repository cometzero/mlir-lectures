# MLIR NPU Compiler - Lecture 19 Quiz

1. Why is a tensor descriptor more than a pointer?
2. What is the difference between logical shape and physical stride?
3. Why should command buffers include an ABI version and target ID?
4. Explain the difference between a command-internal barrier and a host-visible fence.
5. When is EmitC a better codegen path than LLVM IR for edge deployment?
6. Why does an AOT binary command buffer need a relocation table?
7. Give two examples of negative ABI tests for an NPU runtime compiler.
8. What fields should be included in a runtime capability query?
9. Why should runtime not infer layout from only tensor rank and shape?
10. What is the purpose of a descriptor table in a command buffer?

## Suggested answers

1. It contains metadata such as shape, stride, dtype, layout, memory space, quantization, and flags.
2. Shape is logical tensor extent; stride is physical memory address traversal.
3. To detect incompatible runtime/firmware/compiler artifacts.
4. Barrier orders device commands; fence exposes completion to the host.
5. When static C/C++ generation and cross compilation are more important than LLVM-level optimization or JIT.
6. Device addresses are not known until runtime allocation/load time.
7. Unsupported layout, missing fence, illegal memory space transfer, unaligned descriptor, unsupported dtype.
8. Supported dtypes, tile sizes, memory sizes, max command buffer size, firmware version, feature flags.
9. Packed and blocked layouts need explicit physical addressing metadata.
10. To let commands reference tensors by stable indices rather than raw pointers.
