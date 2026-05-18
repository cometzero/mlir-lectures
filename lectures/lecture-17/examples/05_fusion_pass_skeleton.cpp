// Lecture 17: NPU graph fusion pass skeleton. Pseudo C++.
// Focus: match MatMul + Bias + GELU and attach layout contract.

struct FuseMlpEpiloguePass
    : public PassWrapper<FuseMlpEpiloguePass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseMlpEpiloguePass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect,
                    arith::ArithDialect, npu::NpuGraphDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<MatMulBiasGeluToNpuEpilogue>(&getContext(), getCostModel());
    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns), config)))
      signalPassFailure();
  }

  NpuFusionCostModel getCostModel() const {
    return NpuFusionCostModel{/*sramBytes=*/2 * 1024 * 1024,
                              /*nativeM=*/16, /*nativeN=*/16, /*nativeK=*/32};
  }
};

struct MatMulBiasGeluToNpuEpilogue : OpRewritePattern<npu::GeluOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(npu::GeluOp gelu, PatternRewriter &rewriter) const override {
    auto bias = gelu.getInput().getDefiningOp<linalg::GenericOp>();
    if (!bias || !isBiasAdd(bias)) return failure();

    auto matmul = getProducerMatmul(bias);
    if (!matmul || !hasOneUseThrough(bias, gelu)) return failure();

    FusionContract c;
    if (failed(checkShapeDTypeLayoutQuant(matmul, bias, gelu, c))) return failure();
    if (!costModel.isProfitable(c)) return failure();

    auto fused = rewriter.create<npu::MatMulEpilogueOp>(
        gelu.getLoc(), gelu.getResult().getType(), matmul.getInputs()[0],
        matmul.getInputs()[1], getBiasOperand(bias),
        rewriter.getStringAttr("bias_gelu_tanh"), c.toAttrDictionary(rewriter));
    rewriter.replaceOp(gelu, fused.getResult());
    return success();
  }
};
