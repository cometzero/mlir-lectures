# NPU Buffer Lifetime Table Template

| Buffer | Producer | Last use | Size bytes | Memory space | Reuse candidate | Hazard |
|---|---|---:|---:|---|---|---|
| A_tile_ping | dma_load_A | matmul_k0 | 16384 | SRAM | A_tile_next? | ping/pong overlap |
| B_tile_ping | dma_load_B | matmul_k0 | 16384 | SRAM | B_tile_next? | ping/pong overlap |
| ACC_tile | zero_acc | store_C | 65536 | ACCUM | next output tile | reduction live range |
| C_tile | requant | dma_store_C | 16384 | SRAM/DRAM | output staging | store completion |
