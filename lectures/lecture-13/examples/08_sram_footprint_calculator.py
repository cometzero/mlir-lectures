#!/usr/bin/env python3
Mt, Nt, Kt = 64, 64, 128
A = Mt * Kt * 2
B = Kt * Nt * 2
C_acc = Mt * Nt * 4
C_stage = Mt * Nt * 1
SRAM = 2 * (A + B) + C_stage
for name, val in [("A_tile", A), ("B_tile", B), ("C_acc", C_acc), ("C_stage", C_stage), ("SRAM_total", SRAM)]:
    print(f"{name}: {val} bytes ({val/1024:.1f} KiB)")
