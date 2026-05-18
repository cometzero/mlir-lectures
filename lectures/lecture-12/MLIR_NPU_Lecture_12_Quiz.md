# MLIR NPU Lecture 12 Quiz

## Multiple choice

1. Bufferization primarily converts which semantic boundary?
   - A. LLVM IR to object code
   - B. Tensor semantics to memref/storage semantics
   - C. Python AST to MLIR
   - D. Verilog to bitstream

2. Which property makes DPS helpful for One-Shot Bufferize?
   - A. It always removes loops
   - B. It provides a candidate destination buffer for each result
   - C. It forces all tensors into DRAM
   - D. It eliminates all aliases

3. In an NPU compiler, unexpected memref.copy usually affects:
   - A. only parser speed
   - B. only file size
   - C. bandwidth, latency, and SRAM/DRAM traffic
   - D. source-level comments

## Short answer

4. Explain RaW conflict in one sentence.

5. Why should bufferization generally happen after tensor-level fusion/tiling?

6. Give two fields that an NPU runtime tensor descriptor needs after function-boundary bufferization.

## Design question

7. A MatMul epilogue creates `%C1 = matmul(... outs(%C0))` and `%C2 = relu(%C1 outs(%C1))`. Under what conditions can `%C2` reuse the same physical buffer as `%C1`?
