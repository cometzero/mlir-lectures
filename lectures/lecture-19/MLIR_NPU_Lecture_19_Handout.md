# MLIR NPU Compiler - Lecture 19 Handout

## Topic
Runtime / ABI / Codegen: command buffer and tensor descriptor ABI.

## Core idea
Lecture 19 is the point where compiler IR becomes a product artifact. The final runtime-visible layer should not depend on unstable compiler-only assumptions. The ABI must be versioned, testable, and compatible with runtime/firmware updates.

## Minimal lowering path

```text
npu_kernel.matmul/dma/barrier
  -> npu_rt.tensor_desc / npu_rt.command_buffer
  -> descriptor table + relocation table + command packets
  -> LLVM host stub / EmitC static stub / binary artifact
  -> runtime queue submit + fence
```

## Required outputs

1. Tensor descriptor ABI table.
2. Command buffer packet schema.
3. Runtime C API sketch.
4. Codegen path decision: LLVM IR, EmitC, binary, or hybrid.
5. Regression test matrix.

## NPU-specific advice

- Keep runtime-visible schema stable.
- Never use compiler-internal attributes as implicit ABI.
- Explicitly encode shape, stride, dtype, layout, memory space, quantization, and dependency tokens.
- Separate BARRIER from FENCE.
- Always include ABI version and target feature list.
