#!/usr/bin/env python3
import math

def required_bits(k: int, max_abs_a: int, max_abs_w: int, bias_margin: int = 0) -> int:
    bound = k * max_abs_a * max_abs_w + bias_margin
    return 1 + math.ceil(math.log2(bound + 1)), bound

cases = [
    (4096, 127, 127, 1 << 24, "int8 symmetric"),
    (4096, 255, 255, 1 << 24, "affine uint8 correction bound"),
    (16384, 127, 127, 1 << 24, "large K int8"),
    (4096, 7, 7, 1 << 20, "int4 symmetric"),
]

for k, a, w, bias, name in cases:
    bits, bound = required_bits(k, a, w, bias)
    print(f"{name:32s} K={k:6d} bound={bound:14d} required_bits={bits}")
