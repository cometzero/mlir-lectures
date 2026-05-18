# Transform Dialect Debugging Notes

Recommended debug loop:
1. Start with one small payload function.
2. Run only match + print handles.
3. Add one transform operation at a time.
4. After tile/fuse/vectorize, use the returned handles only.
5. Add `-transform-dialect-check-uses` in CI to catch may-use-after-free.
6. Keep negative tests for unsupported shapes and layouts.

Useful commands:
```bash
mlir-opt input.mlir --transform-interpreter --mlir-print-ir-after-change
mlir-opt input.mlir --transform-dialect-check-uses
mlir-opt input.mlir --transform-infer-effects
```
