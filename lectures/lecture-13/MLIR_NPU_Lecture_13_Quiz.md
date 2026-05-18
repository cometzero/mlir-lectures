# MLIR NPU Compiler - Lecture 13 Quiz

1. `memref<32x?xf16, strided<[?,1], offset:?>, 1>`에서 runtime descriptor로 남아야 하는 metadata를 쓰세요.
2. `memref.subview`가 copy가 아니라 view인 이유를 offsets/sizes/strides 관점에서 설명하세요.
3. memory space를 raw integer보다 custom attribute로 표현하면 좋은 이유를 세 가지 쓰세요.
4. `A_tile[64x128xf16]`, `B_tile[128x64xf16]`, `C_acc[64x64xi32]`의 byte footprint를 계산하세요.
5. `memref.memory_space_cast`와 DMA copy의 차이를 설명하세요.
6. DRAM/SRAM/ACCUM 각각의 verifier rule을 하나씩 쓰세요.

## Answer Key

1. dynamic dimension, dynamic stride, dynamic offset, base/aligned pointer, dtype, memory space, rank.
2. base buffer를 공유하고 view metadata만 바꾼다. result offset/strides는 source strides와 operation offsets/sizes/strides로 계산된다.
3. readability, verifier rule encoding, bank/scope/alignment metadata, ABI evolution.
4. A=16 KiB, B=16 KiB, C=16 KiB, total=48 KiB before padding/tags.
5. cast는 같은 underlying memory를 다른 memref type으로 alias할 수 있는 operation이고 DMA copy는 실제 data movement다.
6. DRAM: DMA reachable/aligned. SRAM: capacity/bank/lifetime valid. ACCUM: legal producer/consumer and no illegal escape.
