#!/usr/bin/env python3
"""Compute fixed-point multiplier/shift for Lecture 16.

This is a teaching approximation. Production compilers should match
hardware rounding exactly and include exhaustive edge tests.
"""
import math

def choose_multiplier_shift(r: float, max_bits: int = 31, max_shift: int = 62):
    assert r > 0.0
    best = None
    max_m = (1 << max_bits) - 1
    for s in range(max_shift + 1):
        m = int(round(r * (1 << s)))
        if m <= 0 or m > max_m:
            continue
        err = abs(r - m / (1 << s))
        cand = (err, m, s)
        if best is None or cand < best:
            best = cand
    if best is None:
        raise ValueError("ratio cannot be represented with constraints")
    err, m, s = best
    return m, s, err

if __name__ == "__main__":
    cases = [
        (0.03125, 0.015625, 0.0078125),
        (0.02, 0.03, 0.01),
        (0.007, 0.018, 0.004),
    ]
    for sa, sw, sy in cases:
        r = sa * sw / sy
        m, s, err = choose_multiplier_shift(r)
        print(f"sa={sa:.8f} sw={sw:.8f} sy={sy:.8f} r={r:.10f} -> M={m} S={s} err={err:.3e}")
