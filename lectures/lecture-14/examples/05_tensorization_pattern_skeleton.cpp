// Lecture 14 - C++ pseudo skeleton: vector.contract -> npu_kernel.matmul
// This is intentionally incomplete and designed for lecture discussion.

struct TensorizeMatMulContractPattern
    : public OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern<vector::ContractionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp op,
                                PatternRewriter &rewriter) const override {
    // 1. Match iterator types: [parallel, parallel, reduction].
    if (!isMatmulIteratorTypes(op.getIteratorTypesArray()))
      return rewriter.notifyMatchFailure(op, "not a matmul iterator pattern");

    // 2. Match indexing maps: A(m,k), B(k,n), C(m,n).
    if (!isSupportedMatmulMaps(op.getIndexingMapsArray()))
      return rewriter.notifyMatchFailure(op, "unsupported indexing maps/layout");

    // 3. Extract vector shapes and element types.
    VectorType lhsTy = cast<VectorType>(op.getLhsType());
    VectorType rhsTy = cast<VectorType>(op.getRhsType());
    VectorType accTy = cast<VectorType>(op.getAccType());

    // 4. Query target tile shape from NPU target description.
    // Example: M16N16K32 int8 -> int32.
    if (!targetSupportsM16N16K32I8(lhsTy, rhsTy, accTy))
      return rewriter.notifyMatchFailure(op, "not a native NPU tile");

    // 5. Rewrite to target-specific op.
    auto newOp = rewriter.create<npu_kernel::MatMulOp>(
        op.getLoc(), accTy, op.getLhs(), op.getRhs(), op.getAcc(),
        /*m=*/16, /*n=*/16, /*k=*/32,
        /*lhs_layout=*/"mk", /*rhs_layout=*/"kn", /*acc_type=*/"i32");
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};
