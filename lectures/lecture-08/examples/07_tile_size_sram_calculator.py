#!/usr/bin/env python3
# Lecture 08 helper: estimate NPU tile working set.

def footprint(TM, TN, TK, input_bytes=1, acc_bytes=4, double_buffer=True):
    a = TM * TK * input_bytes
    b = TK * TN * input_bytes
    acc = TM * TN * acc_bytes
    total = (2 * (a + b) if double_buffer else (a + b)) + acc
    return {"A_bytes": a, "B_bytes": b, "Acc_bytes": acc, "Total_bytes": total}

if __name__ == "__main__":
    for cfg in [(32,64,64), (64,64,64), (32,128,64)]:
        print(cfg, footprint(*cfg))
