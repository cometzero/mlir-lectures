// Lecture 13 - Subview tile buffer example
func.func @tile_view(%A: memref<1024x1024xf16, strided<[1024, 1], offset: 0>, 0>, %m0: index, %k0: index) {
  %tile = memref.subview %A[%m0, %k0] [64, 128] [1, 1]
    : memref<1024x1024xf16, strided<[1024, 1], offset: 0>, 0>
      to memref<64x128xf16, strided<[1024, 1], offset: ?>, 0>
  return
}
