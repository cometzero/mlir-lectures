// Lecture 11 - ConversionPattern skeleton.
// Key rule: use adaptor operands, not original operands, after type conversion.

struct LowerExactGeluPattern : public OpConversionPattern<npu_graph::GeluExactOp> {
  using OpConversionPattern<npu_graph::GeluExactOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(npu_graph::GeluExactOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto input = adaptor.getInput();
    auto type = dyn_cast<RankedTensorType>(input.getType());
    if (!type || !type.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "GELU lowering requires static shape");

    if (!isSupportedElementType(type.getElementType()))
      return rewriter.notifyMatchFailure(op, "unsupported GELU element type");

    // Policy decision: exact erf is unsupported; lower to NPU LUT approximation.
    auto lut = getOrCreateGeluLUT(op, rewriter, type.getElementType());
    auto result = rewriter.create<npu_graph::ActivationLUTOp>(
        op.getLoc(), type, input, lut,
        rewriter.getStringAttr("gelu"),
        rewriter.getF64FloatAttr(1.0e-3));

    rewriter.replaceOp(op, result.getResult());
    return success();
  }
};
