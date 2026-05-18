# Dialect Conversion Checklist for NPU Compiler

## ConversionTarget
- [ ] List legal dialects after this boundary.
- [ ] Mark high-level source ops illegal.
- [ ] Add dynamic legality for matmul/conv/reduce/layout-sensitive ops.
- [ ] Decide partial vs full conversion mode.

## ConversionPattern
- [ ] Root op selected intentionally.
- [ ] Uses adaptor operands after type conversion.
- [ ] Checks shape, dtype, layout, quantization, target feature bits.
- [ ] Emits legal target ops only.
- [ ] Emits clear diagnostics for unsupported cases.

## TypeConverter
- [ ] Identity conversion for already legal types.
- [ ] Tensor/memref/layout/memory-space conversion policy documented.
- [ ] Region signature conversion considered.
- [ ] Materialization ops minimized and tested.

## Tests
- [ ] Positive FileCheck test.
- [ ] Negative diagnostic test.
- [ ] Dynamic legality edge cases.
- [ ] Numerical tolerance test for approximations.
- [ ] SRAM/cycle model check for accepted target ops.
