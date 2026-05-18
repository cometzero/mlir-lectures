# NPU Runtime Descriptor Layout Draft

```c
struct NpuTensorDesc {
  uint64_t handle_or_phys_addr;
  uint32_t byte_offset;
  uint16_t rank, dtype, memory_space, layout_id;
  uint32_t alignment, flags;
  int64_t sizes[MAX_RANK];
  int64_t byte_strides[MAX_RANK];
};
```

DRAM descriptors may be host-visible; SRAM descriptors are usually command-buffer-local; ACCUM descriptors may be internal compiler artifacts.
