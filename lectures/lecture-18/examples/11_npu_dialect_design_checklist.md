# NPU Dialect Design Checklist

- [ ] Dialect boundary is clear: graph vs kernel vs runtime.
- [ ] `npu.matmul` tile shape/layout/dtype/accumulator contract is explicit.
- [ ] `npu.dma` source/destination memory space and size unit are explicit.
- [ ] `npu.barrier` token list and scope are explicit.
- [ ] Verifier rejects illegal target contracts, not merely unprofitable schedules.
- [ ] Parser/printer tests cover custom and generic assembly.
- [ ] Negative tests use `-verify-diagnostics`.
- [ ] Lowering tests cover `vector.contract -> npu.matmul` and DMA/barrier materialization.
