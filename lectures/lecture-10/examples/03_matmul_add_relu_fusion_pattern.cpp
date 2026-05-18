// Lecture 10 - pseudo C++ pattern skeleton.
// Purpose: show match discipline and PatternRewriter usage, not a drop-in build file.

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

struct FuseMatMulBiasReluPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  void initialize() {
    setDebugName("FuseMatMulBiasReluPattern");
    addDebugLabels("NPUFusion", "EpilogueFusion");
  }

  LogicalResult matchAndRewrite(linalg::GenericOp relu,
                                PatternRewriter &rewriter) const override {
    // 1) Match root as ReLU-like generic op.
    if (!isReluGeneric(relu))
      return rewriter.notifyMatchFailure(relu, "root is not ReLU generic");

    // 2) Match producer Add.
    auto add = getSingleProducerOfType<linalg::GenericOp>(relu.getDpsInputOperand(0));
    if (!add || !isAddBiasGeneric(add))
      return rewriter.notifyMatchFailure(relu, "producer is not AddBias");

    // 3) Match MatMul input and bias operand.
    Value matmulResult;
    Value bias;
    if (failed(classifyAddInputs(add, matmulResult, bias)))
      return rewriter.notifyMatchFailure(add, "cannot classify matmul and bias");

    auto matmul = getSingleProducerOfType<linalg::GenericOp>(matmulResult);
    if (!matmul || !isMatMulGeneric(matmul))
      return rewriter.notifyMatchFailure(add, "input is not matmul");

    // 4) Prove safety and target legality.
    if (!matmul->hasOneUse() || !add->hasOneUse())
      return rewriter.notifyMatchFailure(relu, "producer result is shared");
    if (failed(checkShapeDTypeLayoutForTarget(matmul, add, relu, bias)))
      return rewriter.notifyMatchFailure(relu, "NPU fusion contract failed");

    // 5) Rewrite only after all checks have succeeded.
    Location loc = rewriter.getFusedLoc({matmul.getLoc(), add.getLoc(), relu.getLoc()},
                                        "npu.matmul_bias_relu");
    Value lhs = matmul.getDpsInputOperand(0)->get();
    Value rhs = matmul.getDpsInputOperand(1)->get();
    Type resultType = relu.getResult(0).getType();

    auto fused = rewriter.create<npu_graph::MatmulEpilogueOp>(
        loc, resultType, lhs, rhs, bias,
        /*activation=*/rewriter.getStringAttr("relu"));

    rewriter.replaceOp(relu, fused.getResults());
    rewriter.eraseOp(add);
    rewriter.eraseOp(matmul);
    return success();
  }
};

} // namespace
