# Lecture 01 Example: MatMul+Bias+ReLU Lowering Notes

Assumptions: M=128, K=256, N=512, PE tile=64x64, K chunk=128, SRAM=2MB.

Pipeline draft:
1. Import from StableHLO/TOSA.
2. Legalize unsupported ops.
3. Canonicalize broadcast and constants.
4. Fuse dot_general + bias add + ReLU.
5. Tile M/N/K to 64/64/128.
6. Bufferize tensors into DRAM/SRAM/ACC memory spaces.
7. Schedule DMA load, compute, post-op, store with barriers.
8. Emit tensor descriptors and command buffer.

Correctness: shape/type checks, tile coverage, golden numerical output.
Performance: PE utilization, DRAM bytes per tile, DMA/compute overlap, latency.
