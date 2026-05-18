# Capstone Rubric and Review Checklist

## Rubric
| Category | Weight | Excellent Criteria |
|---|---:|---|
| Correctness | 25 | Legal IR contracts, verifier invariants, tests cover boundaries |
| Architecture Fit | 20 | NPU memory hierarchy and native tile constraints are explicit |
| Optimization | 20 | Fusion/layout/quantization decisions are justified by a cost model |
| Implementation Plan | 15 | Pass pipeline and dialect conversion boundaries are realistic |
| Validation | 10 | Positive/negative/FileCheck/golden tests are included |
| Presentation | 10 | Tradeoffs are clear and design is explainable in 10 minutes |

## Review Checklist
- [ ] The input IR contract is explicit.
- [ ] Legal op set includes dynamic legality conditions.
- [ ] Unsupported ops have fallback or decomposition policy.
- [ ] Schedule decisions fit target tile and memory assumptions.
- [ ] Runtime ABI includes descriptor layout and command packet schema.
- [ ] Tests include at least one negative verifier case.
- [ ] Cost model includes SRAM footprint and traffic estimate.
