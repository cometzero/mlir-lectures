# MLIR NPU Lecture 16 Handout - Quantization

## 핵심 수식

```text
q = clamp(round(x / scale) + zero_point)
x_hat = (q - zero_point) * scale
acc_i32 = sum_k (qa - za) * (qw - zw)
qy = clamp(round(acc_i32 * scale_a * scale_w / scale_y) + zy)
```

## MLIR에서 봐야 할 것

- `!quant.uniform<i8:f32, scale:zero_point>`: stored integer와 expressed float 사이의 mapping.
- `quant.qcast`: expressed type에서 quantized type으로 변환.
- `quant.dcast`: quantized type에서 expressed type으로 변환.
- `quant.scast`: quantized type과 storage integer type 사이의 bit reinterpretation.
- `tosa.rescale`: integer multiply/add/shift로 requantization 표현.

## NPU compiler checklist

1. qparams: scale > 0, zero point range, symmetric/asymmetric policy.
2. granularity: per-tensor, per-channel, blockwise/group-wise.
3. accumulator: K와 operand range에 대한 overflow check.
4. requant: multiplier/shift bit-width, rounding, saturation.
5. INT4: nibble packing, group_size, scale layout, DMA alignment.
6. runtime ABI: qparams buffer, packed weight layout, descriptor fields.

## Recommended pipeline

```text
StableHLO/TOSA/Linalg
  -> import calibration/QAT qparams
  -> insert/recognize QDQ
  -> propagate/canonicalize QDQ
  -> legalize to integer NPU graph
  -> materialize multiplier/shift
  -> verify accumulator and saturation
  -> pack weights
  -> emit npu_kernel.qmatmul/qconv
  -> emit runtime descriptors
```

## 실습 산출물

- INT8 MatMul quantization equation annotation
- `!quant.uniform` type examples
- `tosa.rescale` pseudo IR
- accumulator width table
- INT4 packing contract
- FileCheck tests
