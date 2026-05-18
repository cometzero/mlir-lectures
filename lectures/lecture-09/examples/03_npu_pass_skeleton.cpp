// Pseudo C++ skeleton for Lecture 09. Names are illustrative.
#include "mlir/Pass/Pass.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

using namespace mlir;

namespace npu {
class NpuGraphDialect;
class NpuKernelDialect;
} // namespace npu

namespace {
struct FuseMatmulEpiloguePass
    : public PassWrapper<FuseMatmulEpiloguePass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseMatmulEpiloguePass)

  Option<bool> enableRelu{*this, "enable-relu",
                          llvm::cl::desc("Fuse ReLU epilogue"),
                          llvm::cl::init(true)};
  Statistic numFused{this, "num-fused", "Number of matmul epilogues fused"};

  StringRef getArgument() const final { return "npu-fuse-matmul-epilogue"; }
  StringRef getDescription() const final {
    return "Fuse MatMul + Bias + ReLU into an NPU graph candidate op";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    // TODO: registry.insert<npu::NpuGraphDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    // TODO: query analyses if needed.
    // auto &dominance = getAnalysis<DominanceInfo>();

    bool sawIllegalPattern = false;
    func.walk([&](linalg::MatmulOp matmul) {
      // TODO: match bias + activation users.
      // TODO: replace with npu_graph.matmul {epilogue = "relu"}.
      // ++numFused;
    });

    if (sawIllegalPattern) {
      func.emitError("illegal MatMul epilogue pattern for target NPU");
      return signalPassFailure();
    }

    // TODO: mark analyses preserved only if invariant is truly unchanged.
    // markAnalysesPreserved<DominanceInfo>();
  }
};
} // namespace

std::unique_ptr<Pass> createFuseMatmulEpiloguePass() {
  return std::make_unique<FuseMatmulEpiloguePass>();
}
