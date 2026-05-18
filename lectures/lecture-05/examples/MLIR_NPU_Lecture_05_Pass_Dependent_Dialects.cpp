// MLIR_NPU_Lecture_05_Pass_Dependent_Dialects.cpp
// Educational pseudo C++ skeleton for pass dialect dependencies.

#include "mlir/Pass/Pass.h"
#include "mlir/IR/DialectRegistry.h"

using namespace mlir;

namespace npu::graph { class NPUGraphDialect; }
namespace npu::kernel { class NPUKernelDialect; }
namespace npu::rt { class NPURuntimeDialect; }

namespace {
struct LowerGraphToNPUKernelPass
    : public PassWrapper<LowerGraphToNPUKernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerGraphToNPUKernelPass)

  StringRef getArgument() const final { return "lower-npu-graph-to-kernel"; }
  StringRef getDescription() const final {
    return "Lower target-aware graph dialect operations to NPU kernel operations";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<
        npu::kernel::NPUKernelDialect,
        memref::MemRefDialect,
        scf::SCFDialect,
        arith::ArithDialect>();
  }

  void runOnOperation() override {
    // 1. Define ConversionTarget.
    // 2. Mark graph ops illegal and kernel/memref/scf ops legal.
    // 3. Populate RewritePatternSet.
    // 4. Apply partial/full conversion.
    // 5. Signal pass failure if legality cannot be satisfied.
  }
};
} // namespace
