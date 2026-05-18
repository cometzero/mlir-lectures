#!/usr/bin/env python3
"""Lecture 17: simple MLP fusion traffic estimator.

This intentionally ignores cache effects. Use it to compare unfused vs fused
intermediate writes for a Transformer MLP tile.
"""
from dataclasses import dataclass

@dataclass
class MLPShape:
    batch_tokens: int = 128
    hidden: int = 4096
    expansion: int = 4
    bytes_per_elem: int = 2  # fp16/bf16

    @property
    def expanded(self) -> int:
        return self.hidden * self.expansion

    def tensor_bytes(self, cols: int) -> int:
        return self.batch_tokens * cols * self.bytes_per_elem


def estimate(shape: MLPShape) -> None:
    h4 = shape.tensor_bytes(shape.expanded)
    h = shape.tensor_bytes(shape.hidden)
    # Simplified: GEMM1 output, bias output, activation output all hit DRAM when unfused.
    unfused_intermediate_writes = 3 * h4
    fused_epilogue_writes = 1 * h4
    saved = unfused_intermediate_writes - fused_epilogue_writes
    print(f"tokens={shape.batch_tokens}, H={shape.hidden}, expansion={shape.expansion}x")
    print(f"expanded activation bytes: {h4/1024/1024:.2f} MiB")
    print(f"unfused epilogue writes:  {unfused_intermediate_writes/1024/1024:.2f} MiB")
    print(f"fused epilogue writes:    {fused_epilogue_writes/1024/1024:.2f} MiB")
    print(f"saved writes:             {saved/1024/1024:.2f} MiB")
    print(f"final output bytes:       {h/1024/1024:.2f} MiB")

if __name__ == "__main__":
    estimate(MLPShape())
