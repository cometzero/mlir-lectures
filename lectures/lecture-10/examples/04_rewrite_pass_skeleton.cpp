// Lecture 10 - pass skeleton to run the pattern set greedily.

#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {
struct NPUFuseEpiloguePass
    : public PassWrapper<NPUFuseEpiloguePass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NPUFuseEpiloguePass)

  StringRef getArgument() const override { return "npu-fuse-epilogue"; }
  StringRef getDescription() const override {
    return "Fuse MatMul + Bias + ReLU into an NPU graph epilogue op";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, arith::ArithDialect,
                    npu_graph::NPUGraphDialect>();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<FuseMatMulBiasReluPattern>(ctx);

    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.maxIterations = 10;

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns), config)))
      signalPassFailure();
  }
};
} // namespace
