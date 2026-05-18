// Lecture 08 - Example 04: double-buffered DMA schedule pseudo IR.

func.func @double_buffered_matmul_schedule(
    %A: memref<128x256xi8>,
    %B: memref<256x128xi8>,
    %C: memref<128x128xi32>) {
  %c0 = arith.constant 0 : index
  %K = arith.constant 256 : index
  %TK = arith.constant 64 : index

  // Preload k0 into buffer 0.
  "npu.dma.start"(%A, %B, %c0) {buffer = 0 : i32} : (memref<128x256xi8>, memref<256x128xi8>, index) -> ()
  "npu.dma.wait"() {buffer = 0 : i32} : () -> ()

  scf.for %k0 = %c0 to %K step %TK {
    "npu.dma.start_next_if_any"(%A, %B, %k0) : (memref<128x256xi8>, memref<256x128xi8>, index) -> ()
    "npu.matmul.consume_current_buffer"(%k0) : (index) -> ()
    "npu.dma.wait_next_if_any"(%k0) : (index) -> ()
  }

  "npu.dma.store"(%C) : (memref<128x128xi32>) -> ()
  return
}
