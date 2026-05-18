// Pseudo TableGen pass specification for Lecture 09.
include "mlir/Pass/PassBase.td"

def NpuFuseMatmulEpilogue : Pass<"npu-fuse-matmul-epilogue", "mlir::func::FuncOp"> {
  let summary = "Fuse MatMul + Bias + Activation into an NPU graph op";
  let description = [{
    Matches a linalg.matmul producer and epilogue consumers such as bias add and ReLU.
    Emits a target-aware NPU graph candidate op while preserving high-level tensor semantics.
  }];
  let dependentDialects = ["npu::NpuGraphDialect"];
  let options = [
    Option<"enableRelu", "enable-relu", "bool", "true", "Fuse ReLU epilogue">,
    Option<"maxEpilogueOps", "max-epilogue-ops", "unsigned", "4", "Maximum epilogue ops to fuse">
  ];
  let statistics = [
    Statistic<"numFused", "num-fused", "Number of fused matmul epilogues">,
    Statistic<"numRejected", "num-rejected", "Number of rejected fusion candidates">
  ];
}
