// Lecture 08 - Example 01: scf.for anatomy and loop-carried value.

func.func @scf_for_basic(%A: memref<128xf32>, %B: memref<128xf32>) {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index

  scf.for %i = %c0 to %c128 step %c1 {
    %a = memref.load %A[%i] : memref<128xf32>
    memref.store %a, %B[%i] : memref<128xf32>
  }
  return
}

func.func @scf_for_reduce(%A: memref<128xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32
  %sum = scf.for %i = %c0 to %c128 step %c1 iter_args(%acc = %zero) -> (f32) {
    %a = memref.load %A[%i] : memref<128xf32>
    %next = arith.addf %acc, %a : f32
    scf.yield %next : f32
  }
  return %sum : f32
}
