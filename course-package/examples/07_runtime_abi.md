# Runtime ABI Sketch

```c
struct TensorDesc {
  uint64_t device_addr;
  uint32_t rank;
  int64_t shape[6];
  int64_t stride[6];
  uint32_t dtype;
  uint32_t layout;
  uint32_t memory_space;
};

struct NpuCommandHeader {
  uint16_t opcode;
  uint16_t flags;
  uint32_t payload_bytes;
  uint64_t payload_addr;
};
```

APIs:

```c
npu_module_t* npu_load_module(const void* blob, size_t bytes);
int npu_bind_tensor(npu_module_t*, const char* name, const TensorDesc* desc);
int npu_submit(npu_module_t*, npu_stream_t* stream);
int npu_sync(npu_stream_t* stream);
int npu_read_profile(npu_module_t*, NpuProfile* out);
```
