# Lecture 11 Quick Reference - Dialect Conversion

## Minimal pass shape
```cpp
ConversionTarget target(ctx);
target.addLegalDialect<npu_graph::NPUGraphDialect>();
target.addIllegalOp<npu_graph::GeluExactOp>();
target.addDynamicallyLegalOp<npu_graph::MatMulOp>(predicate);
RewritePatternSet patterns(&ctx);
patterns.add<LowerUnsupportedOps>(typeConverter, &ctx);
if (failed(applyPartialConversion(module, target, std::move(patterns))))
  signalPassFailure();
```

## Dynamic legality checklist
- Static/dynamic shape
- MMA tile compatibility
- DType and accumulator width
- Layout/packing state
- Quantization scale/zero-point policy
- SRAM tile fit
- Epilogue/fusion support

## Unsupported op policy
`direct -> exact decomposition -> approximate with error budget -> explicit fallback -> reject diagnostic`
