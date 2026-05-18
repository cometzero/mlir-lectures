# Quantization Policy Checklist

## QParams

- [ ] scale > 0
- [ ] zero_point within storage range
- [ ] symmetric/asymmetric policy is explicit
- [ ] per-channel scale length matches channel dimension
- [ ] layout transform remaps qparams axis

## Integer Kernel

- [ ] accumulator width is verified
- [ ] zero-point correction is implemented or statically unnecessary
- [ ] bias scale equals input_scale * weight_scale[channel]
- [ ] requant multiplier/shift fit hardware limits
- [ ] rounding mode is specified
- [ ] saturation range matches output quant type

## INT4

- [ ] signedness is specified
- [ ] low/high nibble order is specified
- [ ] K dimension aligned to group_size and native tile
- [ ] qparams layout matches packed weight layout
- [ ] DMA padding does not change logical shape
