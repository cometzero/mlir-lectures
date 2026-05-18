// This file is a planning artifact, not an upstream-valid MLIR file.
// Goal: write a pass pipeline draft before implementing passes.

// Example command line once passes exist:
// mlir-opt input.mlir \
//   --pass-pipeline='builtin.module(
//     canonicalize,cse,
//     func.func(npu-decompose-composites,npu-fuse-matmul-epilogue),
//     npu-infer-layout{target=npu256},
//     npu-tile-linalg{tile-m=64 tile-n=128 tile-k=32},
//     one-shot-bufferize,
//     npu-assign-memory-space{sram-kb=512},
//     npu-insert-dma{double-buffer=true},
//     npu-place-barriers,
//     npu-emit-runtime-abi{abi-version=1}
//   )'

// Checklist table:
// Pass                         Anchor          Creates dialect     Required analysis        Failure condition
// npu-decompose-composites     func.func       arith/tensor/linalg Shape                   unsupported dynamic rank
// npu-fuse-matmul-epilogue     func.func       npu_graph           Dominance, Shape         illegal broadcast/relu
// npu-infer-layout             builtin.module  npu_graph attrs     Shape, TargetInfo        layout conflict
// npu-tile-linalg              func.func       scf/vector          Shape, LayoutInfo        tile footprint exceeds SRAM
// npu-assign-memory-space      builtin.module  memref attrs        Liveness, Alias          allocation cannot fit
// npu-insert-dma               npu.kernel      npu_kernel          DependencyGraph          hazard cannot be resolved
// npu-emit-runtime-abi         builtin.module  npu_rt              SymbolTable              missing descriptor field
