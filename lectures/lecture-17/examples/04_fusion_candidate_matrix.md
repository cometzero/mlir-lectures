# Lecture 17 Fusion Candidate Matrix

| Candidate | Required condition | NPU benefit | Risk / fallback |
|---|---|---|---|
| MatMul + Bias | bias is broadcast over N; dtype supported | remove intermediate write, use epilogue adder | unsupported broadcast or scale axis |
| MatMul + Bias + GELU | activation approximation accepted; accumulator precision defined | fuse epilogue, reduce DRAM traffic | accuracy, LUT/tanh hardware missing |
| MatMul + Bias + ReLU | simple epilogue supported | high benefit, low risk | none unless quant clamp differs |
| SwiGLU elementwise | gate/up tensors available in SRAM or tile-streamed | avoid writing gate/up intermediates | double live range; SRAM overflow |
| Layout propagation through elementwise | op is layout-preserving | avoid repack/copy | incompatible op or per-channel axis mismatch |
| Weight prepack | static weight or cacheable runtime prepack | higher tensor core utilization | prepack memory, cache invalidation |
