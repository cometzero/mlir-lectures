# MLIR NPU Lecture 16 Quick Reference

## Formula

```text
real = (stored - zp) * scale
stored = clamp(round(real / scale) + zp)
acc = sum((qa - za) * (qw - zw))
requant = clamp(round(acc * sa * sw / sy) + zy)
```

## MLIR Quant Types

```mlir
!quant.uniform<i8:f32, 0.03125:-3>
tensor<64x!quant.uniform<i8:f32, 0.03125:-3>>
tensor<128x64x!quant.uniform<i8:f32:1, {0.01, 0.02, 0.03}>>
```

## Operations

| Op | Meaning | NPU compiler use |
|---|---|---|
| `quant.qcast` | float/expressed -> quantized | Q boundary |
| `quant.dcast` | quantized -> float/expressed | DQ boundary |
| `quant.scast` | quantized <-> integer storage | expose storage |
| `tosa.rescale` | integer multiply/add/shift | requant bridge |

## Accumulator Bound

```text
max_abs_acc = K * max_abs_a * max_abs_w + bias_margin
required_bits = 1 + ceil(log2(max_abs_acc + 1))
```

Approximate safe K in signed i32:

| Case | max product | K safe |
|---|---:|---:|
| int8 symmetric [-127,127] | 16129 | ~133k |
| affine uint8 correction bound 255x255 | 65025 | ~33k |
| int4 symmetric [-7,7] | 49 | ~43M |
| int4 unsigned [0,15] | 225 | ~9.5M |

## Common verifier failures

- Per-channel scale length != channel dimension.
- Requant shift > hardware maximum.
- Bias scale != input_scale * weight_scale.
- K dimension not divisible by INT4 group size.
- Rounding/saturation mode omitted from ABI.
- Accumulator may overflow target width.
