// Lecture 13 - memref.dma_start / dma_wait shape
func.func @dma(%src: memref<1024xf16, 0>, %dst: memref<1024xf16, 1>) {
  %num = arith.constant 256 : index
  %idx = arith.constant 0 : index
  %tag = memref.alloc() : memref<1xi32, 2>
  memref.dma_start %src[%idx], %dst[%idx], %num, %tag[%idx]
    : memref<1024xf16, 0>, memref<1024xf16, 1>, memref<1xi32, 2>
  memref.dma_wait %tag[%idx], %num : memref<1xi32, 2>
  return
}
