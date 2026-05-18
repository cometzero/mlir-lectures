// Pseudo NPU memory-space plan after bufferization.
// 0=DRAM, 1=SRAM, 2=ACCUM in this teaching example.

func.func @npu_tiled_matmul(%A: memref<1024x1024xi8, 0>,
                            %B: memref<1024x1024xi8, 0>,
                            %C: memref<1024x1024xi32, 0>) {
  %A_tile = memref.alloc() : memref<128x128xi8, 1>
  %B_tile = memref.alloc() : memref<128x128xi8, 1>
  %ACC = memref.alloc() : memref<128x128xi32, 2>
  "npu.dma_load"(%A, %A_tile) : (memref<1024x1024xi8, 0>, memref<128x128xi8, 1>) -> ()
  "npu.dma_load"(%B, %B_tile) : (memref<1024x1024xi8, 0>, memref<128x128xi8, 1>) -> ()
  "npu.matmul"(%A_tile, %B_tile, %ACC) : (memref<128x128xi8, 1>, memref<128x128xi8, 1>, memref<128x128xi32, 2>) -> ()
  "npu.dma_store"(%ACC, %C) : (memref<128x128xi32, 2>, memref<1024x1024xi32, 0>) -> ()
  memref.dealloc %ACC : memref<128x128xi32, 2>
  memref.dealloc %B_tile : memref<128x128xi8, 1>
  memref.dealloc %A_tile : memref<128x128xi8, 1>
  return
}
