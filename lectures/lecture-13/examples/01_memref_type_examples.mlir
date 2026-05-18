// Lecture 13 - MemRef type examples
#identity = affine_map<(d0, d1) -> (d0, d1)>
#tile_16x32 = affine_map<(m, n) -> (m floordiv 16, n floordiv 32, m mod 16, n mod 32)>

func.func @memref_types(%d0: index, %d1: index) {
  %dram = memref.alloc() : memref<64x128xf16, #identity, 0>
  %sram = memref.alloc() {alignment = 64} : memref<64x128xf16, strided<[160, 1], offset: 0>, 1>
  %dyn = memref.alloc(%d0, %d1) : memref<?x?xf16, strided<[?, ?], offset: ?>, 0>
  %tiled = memref.alloc() : memref<64x128xf16, #tile_16x32, 1>
  return
}
