// Educational pipeline sketch. Exact pass names/options vary by MLIR build.
// Goal: keep linalg-on-tensors until schedule decisions are made.

// 1) Normalize input IR
// mlir-opt input.mlir --canonicalize --cse

// 2) Convert named Linalg ops to generic if a generic-pattern pass needs it
// mlir-opt input.mlir --linalg-generalize-named-ops

// 3) Apply tiling/fusion/vectorization through transform dialect or pass pipeline
// mlir-opt input.mlir --transform-interpreter

// 4) Only then bufferize tensor results into memrefs
// mlir-opt input.mlir --one-shot-bufferize

// 5) Lower to loops/vector/custom NPU dialect
// mlir-opt input.mlir --convert-linalg-to-loops --convert-scf-to-cf
