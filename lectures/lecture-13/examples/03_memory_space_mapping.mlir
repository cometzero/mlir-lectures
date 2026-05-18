// Lecture 13 - Pseudo NPU memory-space mapping
// Convention: 0=DRAM, 1=SRAM, 2=ACCUM
func.func @spaces(%A: memref<1024x1024xf16, 0>, %B: memref<1024x1024xf16, 0>, %C: memref<1024x1024xi8, 0>) {
  %a_tile = memref.alloc() {alignment = 64} : memref<64x128xf16, 1>
  %b_tile = memref.alloc() {alignment = 64} : memref<128x64xf16, 1>
  %c_acc  = memref.alloc() {alignment = 64} : memref<64x64xi32, 2>
  %c_out  = memref.alloc() {alignment = 64} : memref<64x64xi8, 1>
  // npu.dma_load %A -> %a_tile
  // npu.dma_load %B -> %b_tile
  // npu.matmul %a_tile, %b_tile, %c_acc
  // npu.requant %c_acc -> %c_out
  // npu.dma_store %c_out -> %C
  return
}
