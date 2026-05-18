# MLIR/NPU Compiler Quick Reference

## Dialect Ladder

```text
StableHLO/TOSA -> Tensor/Linalg -> Tile/Fuse/Layout -> Bufferization/MemRef -> Vector/Affine/SCF -> NPU Dialect -> Runtime ABI
```

## Metadata to Preserve

| Metadata | Why NPU compiler needs it |
|---|---|
| Static/dynamic shape | tile selection, verifier, command size |
| Layout | DMA burst, bank conflict, NPU MMA mapping |
| Memory space | DRAM/SRAM/ACC placement |
| Quant params | int8/int4 requant and fusion legality |
| Side effects | DMA/barrier scheduling safety |
| Location | profiling and debug feedback |

## Minimal NPU Dialect Op Set

```text
npu.launch
npu.kernel
npu.dma_load / npu.dma_store
npu.matmul / npu.conv2d
npu.elementwise / npu.requant
npu.barrier
npu.return
```

## Pass Pipeline Template

```text
import -> canonicalize/cse -> legalize-to-linalg -> decompose-unsupported -> fuse -> tile/layout -> bufferize -> memory-space-assign -> vectorize/tensorize -> lower-to-npu -> command-buffer-codegen
```
