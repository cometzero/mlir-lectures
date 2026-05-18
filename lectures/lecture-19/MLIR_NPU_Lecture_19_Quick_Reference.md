# MLIR NPU Compiler - Lecture 19 Quick Reference

## Tensor Descriptor Fields

| Field | Meaning | NPU Compiler Check |
|---|---|---|
| magic/version | descriptor schema | ABI compatibility |
| rank/shape | logical tensor extent | shape legality |
| stride | physical address traversal | layout/DMA legality |
| dtype | i8/i4/bf16/fp16/etc | kernel support |
| layout | NHWC/NCHW/blocked/packed | tensorization support |
| memory_space | DRAM/SRAM/ACCUM/CONST | command legality |
| base/offset | runtime address | relocation needed |
| quant | scale/zero point/axis | requant correctness |
| flags | readonly/packed/host_visible | runtime behavior |

## Command Packet Checklist

- command kind
- descriptor indices
- offsets and byte counts
- tile shape or kernel ID
- dependency tokens
- output token
- flags and debug tag
- reserved fields

## Codegen Decision

| Path | Best for | Risk |
|---|---|---|
| LLVM dialect -> LLVM IR | host wrapper, prototype, JIT-like flow | heavier toolchain |
| EmitC -> C/C++ | embedded static build | lower-level optimization limited |
| Binary command buffer | simple runtime submit | harder debug unless manifest exists |
| Hybrid | product deployment | requires schema discipline |

## Runtime API Sketch

```c
typedef struct NpuContext NpuContext;
typedef struct NpuModule NpuModule;
typedef struct NpuQueue NpuQueue;
typedef struct NpuFence NpuFence;

NpuStatus npuInit(uint32_t abi_version);
NpuStatus npuLoadModule(NpuContext*, const void* artifact, size_t size, NpuModule** out);
NpuStatus npuBindTensor(NpuModule*, uint32_t tensor_index, const NpuTensorDesc*, void* device_buffer);
NpuStatus npuSubmit(NpuQueue*, NpuModule*, uint32_t command_buffer_index, NpuFence** out);
NpuStatus npuWait(NpuFence*, uint64_t timeout_ns);
```
