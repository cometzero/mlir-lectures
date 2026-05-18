// Pseudocode: vector.contract -> npu.matmul.
struct ConvertVectorContractToNPUMatmul : OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::ContractionOp op, PatternRewriter &rewriter) const override {
    auto lhs = dyn_cast<VectorType>(op.getLhsType());
    auto rhs = dyn_cast<VectorType>(op.getRhsType());
    auto acc = dyn_cast<VectorType>(op.getAccType());
    if (!isNativeNPUTile(lhs, rhs, acc)) return failure();
    rewriter.replaceOpWithNewOp<npu::MatmulOp>(op, makeAccBufferType(acc), op.getLhs(), op.getRhs(), op.getAcc(), makeTileAttr(16,16,32), makeLayoutAttr("mk"), makeLayoutAttr("kn"));
    return success();
  }
};
