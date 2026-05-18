// Lecture 10 C++ skeleton: MatMul + BiasAdd + ReLU -> npu_graph.matmul.
// This is intentionally partial. Fill in target dialect includes and helpers.

#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

struct EpilogueMatch {
  linalg::GenericOp biasAdd;
  linalg::GenericOp relu;
  Value bias;
};

static bool isSupportedMatmulShape(linalg::MatmulOp op) {
  // TODO: inspect ranked tensor types; reject dynamic rank; apply edge policy.
  return true;
}

static FailureOr<EpilogueMatch> matchBiasAddReluChain(linalg::MatmulOp matmul) {
  // Guard 1: the matmul result must feed only the bias-add candidate.
  if (!matmul->hasOneUse())
    return failure();

  auto biasAdd = dyn_cast<linalg::GenericOp>(*matmul->user_begin());
  if (!biasAdd || !biasAdd->hasOneUse())
    return failure();

  // TODO: verify indexing maps correspond to output + bias broadcast.
  // TODO: verify region body is exactly arith.addf/addi + linalg.yield.

  auto relu = dyn_cast<linalg::GenericOp>(*biasAdd->user_begin());
  if (!relu)
    return failure();

  // TODO: verify region body is maximumf/maxsi/maxui with zero.
  // TODO: reject if dtype/quantization/layout is unsupported.

  EpilogueMatch result;
  result.biasAdd = biasAdd;
  result.relu = relu;
  result.bias = biasAdd.getDpsInputOperand(1)->get();
  return result;
}

struct FuseMatmulBiasReluPattern final : OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp matmul,
                                PatternRewriter &rewriter) const override {
    if (!isSupportedMatmulShape(matmul))
      return rewriter.notifyMatchFailure(matmul, "unsupported NPU matmul shape");

    FailureOr<EpilogueMatch> epilogue = matchBiasAddReluChain(matmul);
    if (failed(epilogue))
      return rewriter.notifyMatchFailure(matmul, "expected matmul -> bias -> relu chain");

    Location loc = epilogue->relu.getLoc();
    Value lhs = matmul.getDpsInputOperand(0)->get();
    Value rhs = matmul.getDpsInputOperand(1)->get();
    Value bias = epilogue->bias;
    Type resultType = epilogue->relu.getResult(0).getType();

    // TODO: replace this pseudo op with your generated npu_graph op class.
    auto fused = rewriter.create<mlir::OperationState>(loc, "npu_graph.matmul");
    (void)lhs; (void)rhs; (void)bias; (void)resultType; (void)fused;

    // In real code, create the op with builder API, then replace relu.
    // auto fused = rewriter.create<npu_graph::MatmulOp>(loc, resultType, lhs, rhs, bias,
    //     rewriter.getStrArrayAttr({"bias", "relu"}));
    // rewriter.replaceOp(epilogue->relu, fused.getResult());
    // rewriter.eraseOp(epilogue->biasAdd);
    // rewriter.eraseOp(matmul);
    return success();
  }
};

} // namespace

void populateNpuFusionPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<FuseMatmulBiasReluPattern>(ctx, PatternBenefit(10));
}
