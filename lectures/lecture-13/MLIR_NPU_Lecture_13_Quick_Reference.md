# Lecture 13 Quick Reference

## MemRef skeleton

```mlir
memref<shape x element-type, layout, memory-space>
```

## Strided address calculation

```text
linear_index = offset + sum_i(index_i * stride_i)
byte_address = base + linear_index * sizeof(element)
```

## Do not confuse

- `memref.subview` changes metadata; it is not a copy.
- `memref.transpose` can be metadata-only for strided memrefs.
- `memref.memory_space_cast` is not DMA.
- ACCUM space may not have C pointer ABI semantics.

## NPU rules

```text
shape/layout/type match kernel contract
SRAM allocation fits capacity/lifetime plan
DMA source/destination spaces are legal
alignment satisfies DMA and vector loads
runtime descriptor has metadata for dynamic dims/strides
```
