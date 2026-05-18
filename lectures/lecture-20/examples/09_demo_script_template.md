# Capstone Demo Script Template

## 1. One-sentence thesis
"Our mini compiler lowers Transformer MLP into fused NPU matmul epilogues and emits a command buffer ABI."

## 2. Show the workload
- Input model graph
- Shape and dtype
- Target NPU assumptions

## 3. Show the pipeline
- Before IR
- After graph legalization/fusion
- After schedule and bufferization
- Final npu_kernel / npu_rt pseudo IR

## 4. Show tests
- FileCheck positive test
- Negative verifier test
- Cost model numbers

## 5. Explain tradeoffs
- What is supported now
- What falls back
- What would be improved next
