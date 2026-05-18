// MLIR_NPU_Lecture_03_Region_Block_Examples.mlir
// Lecture 03: Region and Block examples for AI/NPU compiler study.
// These snippets are intentionally small. Some NPU dialect examples are pseudo IR.

// -----------------------------------------------------------------------------
// Example 1: func.func has a function body region.
// The entry block arguments are function arguments.
// -----------------------------------------------------------------------------
func.func @entry_block_arguments(%A: memref<128x128xf32>, %B: memref<128x128xf32>) {
  %c0 = arith.constant 0 : index
  %v = memref.load %A[%c0, %c0] : memref<128x128xf32>
  memref.store %v, %B[%c0, %c0] : memref<128x128xf32>
  return
}

// -----------------------------------------------------------------------------
// Example 2: scf.for has one body region with one block.
// The induction variable is represented as a block argument of that region.
// -----------------------------------------------------------------------------
func.func @single_block_region(%A: memref<1024xf32>, %B: memref<1024xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c1024 = arith.constant 1024 : index
  scf.for %i = %c0 to %c1024 step %c1 {
    %x = memref.load %A[%i] : memref<1024xf32>
    memref.store %x, %B[%i] : memref<1024xf32>
  }
  return
}

// -----------------------------------------------------------------------------
// Example 3: block argument as merge value.
// scf.execute_region can contain an unstructured CFG.
// -----------------------------------------------------------------------------
func.func @block_argument_merge(%cond: i1) -> i32 {
  %r = scf.execute_region -> i32 {
    cf.cond_br %cond, ^then, ^else

  ^then:
    %a = arith.constant 1 : i32
    cf.br ^merge(%a : i32)

  ^else:
    %b = arith.constant 2 : i32
    cf.br ^merge(%b : i32)

  ^merge(%x : i32):
    scf.yield %x : i32
  }
  return %r : i32
}

// -----------------------------------------------------------------------------
// Example 4: loop-carried variable.
// iter_args initialize additional region block arguments.
// scf.yield passes the next values.
// -----------------------------------------------------------------------------
func.func @loop_carried_sum(%A: memref<1024xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c1024 = arith.constant 1024 : index
  %zero = arith.constant 0.0 : f32
  %sum = scf.for %i = %c0 to %c1024 step %c1
      iter_args(%acc = %zero) -> (f32) {
    %x = memref.load %A[%i] : memref<1024xf32>
    %next = arith.addf %acc, %x : f32
    scf.yield %next : f32
  }
  return %sum : f32
}
