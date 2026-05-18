// Lecture 10 DRR/TableGen skeleton. This is a teaching sketch.
// Use C++ or PDLL when the match requires complex region inspection.

include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"

// Pseudo symbols for explanation only.
def IsLegalNpuMatmulBiasRelu : Constraint<CPred<"isLegalNpuMatmulBiasRelu($_self)">>;
def CreateNpuMatmulBiasRelu : NativeCodeCall<"createNpuMatmulBiasRelu($_builder, $0, $1, $2)">;

// DRR is compact when the DAG is simple, but region-body inspection of
// linalg.generic Add/ReLU usually needs NativeCodeCall or a C++ RewritePattern.
def FuseMatmulBiasRelu : Pat<
  (SomeReluOp (SomeBiasAddOp (linalg_MatmulOp $a, $b), $bias)),
  (CreateNpuMatmulBiasRelu $a, $b, $bias),
  [(IsLegalNpuMatmulBiasRelu $a)]>;
