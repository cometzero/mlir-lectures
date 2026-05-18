# Lecture 20 Capstone Brief

## Goal
Design a mini MLIR-based NPU compiler that lowers a small Transformer MLP or MatMul+Bias+Activation workload into an educational `npu_kernel` / `npu_rt` pseudo dialect and a runtime command buffer ABI.

## Workload Options
1. MatMul + Bias + ReLU / GELU
2. Transformer MLP: `X * W1 + b1 -> GELU -> * W2 + b2`
3. SwiGLU-style MLP: `matmul_a * silu(matmul_b) -> matmul_out`

## Required Final Artifacts
- Mini NPU Compiler Design Spec
- Pass pipeline and IR boundary table
- Legal operation set and unsupported op policy
- Schedule and memory plan
- Custom NPU dialect contract
- Runtime ABI and command buffer schema
- Verification matrix and demo script
