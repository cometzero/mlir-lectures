# Tensor Descriptor ABI Draft

```c
typedef enum {
  NPU_DTYPE_I8 = 1,
  NPU_DTYPE_U8 = 2,
  NPU_DTYPE_I4_PACKED = 3,
  NPU_DTYPE_BF16 = 10,
  NPU_DTYPE_F16 = 11,
  NPU_DTYPE_I32 = 20,
} NpuDType;

typedef enum {
  NPU_MEM_DRAM = 0,
  NPU_MEM_SRAM = 1,
  NPU_MEM_ACCUM = 2,
  NPU_MEM_CONST = 3,
} NpuMemorySpace;

typedef struct {
  uint32_t magic;        // 'NPTD'
  uint16_t abi_major;
  uint16_t abi_minor;
  uint8_t  rank;
  uint8_t  dtype;
  uint8_t  layout;
  uint8_t  memory_space;
  uint64_t base_device_addr;
  uint64_t byte_offset;
  int64_t  shape[8];
  int64_t  stride[8];    // element or byte stride; must be specified by ABI
  float    scale;
  int32_t  zero_point;
  int32_t  quant_axis;   // -1 for per-tensor
  uint32_t flags;
  uint32_t reserved[7];
} NpuTensorDesc;
```

Design rule: keep compiler-only metadata out of this descriptor unless runtime must observe it.
