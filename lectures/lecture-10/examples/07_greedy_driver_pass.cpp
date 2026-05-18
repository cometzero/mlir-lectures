// Lecture 10: minimal pass shell that applies a RewritePatternSet greedily.

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {
struct FuseNpuEpiloguePass
    : public PassWrapper<FuseNpuEpiloguePass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseNpuEpiloguePass)

  StringRef getArgument() const final { return "npu-fuse-matmul-epilogue"; }
  StringRef getDescription() const final {
    return "Fuse MatMul + Bias + ReLU into an NPU graph matmul with epilogue";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    // registry.insert<npu_graph::NpuGraphDialect, linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateNpuFusionPatterns(patterns);

    GreedyRewriteConfig config;
    config.maxIterations = 10;
    config.maxNumRewrites = 1000;

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns), config)))
      signalPassFailure();
  }
};
} // namespace
