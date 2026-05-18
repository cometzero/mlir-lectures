# Layout Propagation Checklist

1. Logical shape is unchanged or mapped through a documented reshape/view rule.
2. Physical layout attribute is updated on every value crossing the boundary.
3. Per-channel quantization axis still refers to the correct logical dimension.
4. Vector/tensorization maps are recomputed after layout propagation.
5. DMA descriptor generation sees the final physical strides/tiles.
6. Materialization points are explicit and testable with FileCheck.
7. Repack cost is accounted in the fusion profitability model.
8. Function ABI layout is documented in the runtime descriptor.
