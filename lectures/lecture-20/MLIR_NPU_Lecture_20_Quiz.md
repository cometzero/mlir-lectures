# MLIR NPU Lecture 20 Quiz

1. Why is a legal op set more than a list of operation names?
2. What information belongs in an operand vs an attribute for `npu.matmul`?
3. Why should bufferization usually happen after tensor-level fusion and scheduling decisions?
4. What are two negative verifier tests for a custom NPU dialect?
5. What fields must be present in a runtime tensor descriptor?
6. How can layout propagation reduce memory traffic?
7. Why is a command buffer ABI a compiler contract, not just a runtime detail?
8. What should a capstone cost model estimate at minimum?

## Answer Key
1. It must include shape, dtype, layout, quantization, and target constraints.
2. Operands carry runtime SSA dependencies; attributes encode compile-time schedule/contract metadata.
3. Tensor value semantics make high-level transformations easier before physical buffer constraints are fixed.
4. Examples: invalid tile size, wrong memory space, missing barrier, unsupported dtype.
5. Base, offset, rank, dims, strides, dtype, layout, memory space, quant metadata.
6. It avoids unnecessary transpose/repack and enables fused producer-consumer layouts.
7. It defines binary compatibility between compiler artifacts and firmware/runtime.
8. SRAM footprint, DRAM traffic, MAC count/cycles, copy count, and fallback penalty.
