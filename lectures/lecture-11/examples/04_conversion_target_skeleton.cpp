// Lecture 11 - ConversionTarget skeleton for NPU graph legalization.
// This is illustrative code; adapt namespace/header names to your project.

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

static bool isLegalMatMulForEdgeNPU(Operation *op) {
  // Pseudo-code. A production pass should use typed ops and verified attrs.
  auto shapeAttr = op->getAttrOfType<ArrayAttr>("shape");
  auto accAttr = op->getAttrOfType<StringAttr>("acc_type");
  auto layoutAttr = op->getAttrOfType<StringAttr>("layout");

  if (!shapeAttr || !accAttr || !layoutAttr) return false;
  // Check: static M/N/K, K % 32 == 0, supported accumulator/layout.
  // Return false to force a conversion pattern or diagnostic.
  return true;
}

struct LegalizeToNPUGraphPass
    : public PassWrapper<LegalizeToNPUGraphPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizeToNPUGraphPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    // registry.insert<npu_graph::NPUGraphDialect, linalg::LinalgDialect,
    //                 tensor::TensorDialect, arith::ArithDialect>();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ModuleOp module = getOperation();

    ConversionTarget target(*ctx);

    // Keep common infrastructure dialects legal.
    // target.addLegalDialect<func::FuncDialect, tensor::TensorDialect,
    //                        arith::ArithDialect>();

    // Target dialect is legal, but selected ops may still need dynamic checks.
    // target.addLegalDialect<npu_graph::NPUGraphDialect>();

    // Example dynamic legality: op is legal only if target constraints hold.
    // target.addDynamicallyLegalOp<npu_graph::MatMulOp>([](npu_graph::MatMulOp op) {
    //   return hasStaticMmaCompatibleShape(op) && hasSupportedDType(op) &&
    //          hasSupportedLayout(op);
    // });

    // Force unsupported high-level ops to be converted.
    // target.addIllegalOp<npu_graph::GeluExactOp>();
    // target.addIllegalOp<npu_graph::LayerNormOp>();

    RewritePatternSet patterns(ctx);
    // patterns.add<LowerMatMulBiasGeluToEpilogue>(ctx);
    // patterns.add<DecomposeLayerNormToPrimitives>(ctx);
    // patterns.add<LowerGatherOrEmitFallback>(ctx);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
