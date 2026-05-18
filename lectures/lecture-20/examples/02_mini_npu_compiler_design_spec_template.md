# Mini NPU Compiler Design Spec Template

## 1. Problem Statement
- Target workload:
- Target NPU assumptions:
- Supported dtypes:
- Required output artifact:

## 2. Input IR Contract
| Field | Decision | Rationale | Test |
|---|---|---|---|
| Input IR | StableHLO / TOSA / Linalg | | |
| Static shapes | Required / Optional | | |
| Dynamic dims | reject / specialize / runtime descriptor | | |
| Layout | NCHW/NHWC/packed | | |
| Quantization | FP16/BF16/INT8/INT4 | | |

## 3. Legal Op Set
| Op | Legal Condition | Lowering | Fallback |
|---|---|---|---|
| matmul | M,N,K multiples of native tile or tail policy exists | npu.matmul | CPU/GPU |
| add bias | broadcast on N dimension | epilogue | materialize |
| GELU | approximate mode supported | LUT / polynomial / fused epilogue | decompose |
| layernorm | axis static, reduction fits | reduction kernel | fallback |

## 4. Pass Pipeline
```text
import -> canonicalize -> decompose -> legalize -> fuse -> layout -> quant ->
tile/schedule -> bufferize -> memory-space -> npu-dialect -> runtime-abi
```

## 5. Schedule and Memory Plan
- Native tile: MxNxK =
- SRAM capacity assumption:
- ACCUM width:
- DMA double buffering policy:
- Edge tile policy:

## 6. Custom NPU Dialect Contract
- `npu_kernel.matmul` operands/results/attrs:
- `npu_kernel.dma` operands/results/attrs:
- `npu_kernel.barrier` token model:
- Verifier invariants:

## 7. Runtime ABI
- Tensor descriptor fields:
- Command buffer header:
- Command packet types:
- Versioning and capability query:

## 8. Validation Plan
- FileCheck positive tests:
- Negative verifier tests:
- Numerical golden tests:
- Cost model checks:
- Demo script:

## 9. Risks and Tradeoffs
| Risk | Impact | Mitigation |
|---|---|---|
| Layout conflict | extra transpose/DMA | layout propagation + materialization policy |
| Quant mismatch | accuracy loss | scale audit and golden tests |
| SRAM overflow | performance cliff | tile search + fallback schedule |
