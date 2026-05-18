# NPU Tensorization Checklist

## Contract

- [ ] Is the op `vector.contract`?
- [ ] Are iterator types exactly matmul-like?
- [ ] Are indexing maps supported by the target layout?
- [ ] Is the combining kind supported?

## Shape

- [ ] M tile supported
- [ ] N tile supported
- [ ] K tile supported
- [ ] Shape is static or target supports dynamic/predicated tile

## DType

- [ ] lhs element type supported
- [ ] rhs element type supported
- [ ] accumulator type supported
- [ ] result type supported
- [ ] requant or cast policy defined

## Memory and Layout

- [ ] lhs memory space legal
- [ ] rhs memory space legal
- [ ] accumulator/result memory space legal
- [ ] alignment and banking constraints met
- [ ] permutation/transpose cost acceptable

## Testing

- [ ] positive FileCheck removes `vector.contract`
- [ ] negative FileCheck does not emit `npu_kernel.matmul`
- [ ] diagnostic explains rejection reason
- [ ] performance counter or model tracks tensorized op count
