# Lecture 15 Schedule Review Checklist

## Transform IR correctness
- [ ] Payload root scope is intentionally selected.
- [ ] Every handle has a clear payload association.
- [ ] Consumed handles are not used later.
- [ ] Silenceable failure paths are documented.

## NPU legality
- [ ] Tile M/N/K matches native compute array or has fallback.
- [ ] SRAM footprint includes double buffering.
- [ ] Accumulator footprint fits target limits.
- [ ] DMA alignment and burst size are valid.
- [ ] Edge tiles have mask/peel/split policy.

## Tests
- [ ] Positive FileCheck test validates loop/vector/npu op shape.
- [ ] Negative test covers unsupported shape/layout.
- [ ] Performance proxy checks tile count and no unexpected copies.
- [ ] Schedule parameters are versioned and reproducible.
