# Mini NPU Compiler Design Spec Template

## 1. Target NPU Summary
- TOPS target:
- Native dtypes:
- Native tile shape:
- SRAM size/banks:
- DRAM bandwidth:

## 2. Input IR and Op Coverage
- StableHLO/TOSA ops:
- Unsupported ops and fallback:

## 3. MLIR Pass Pipeline
```text
import -> canonicalize -> legalize -> fuse -> tile -> bufferize -> memory-plan -> npu-codegen
```

## 4. Memory and Layout Plan
- Memory spaces:
- Layout maps:
- DMA scheduling:

## 5. Custom NPU Dialect
- Ops:
- Verifier constraints:
- Interfaces/traits:

## 6. Runtime ABI
- Tensor descriptor:
- Command buffer:
- Profiling/debug:

## 7. Validation Plan
- FileCheck tests:
- Numerical tolerance:
- Performance counters:
```
