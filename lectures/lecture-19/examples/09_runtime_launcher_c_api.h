#ifndef NPU_RUNTIME_LAUNCHER_C_API_H
#define NPU_RUNTIME_LAUNCHER_C_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NpuContext NpuContext;
typedef struct NpuModule NpuModule;
typedef struct NpuQueue NpuQueue;
typedef struct NpuFence NpuFence;

typedef enum {
  NPU_STATUS_OK = 0,
  NPU_STATUS_ABI_MISMATCH = 1,
  NPU_STATUS_UNSUPPORTED_FEATURE = 2,
  NPU_STATUS_INVALID_DESCRIPTOR = 3,
  NPU_STATUS_INVALID_COMMAND_BUFFER = 4,
  NPU_STATUS_DEVICE_ERROR = 5,
  NPU_STATUS_TIMEOUT = 6,
} NpuStatus;

typedef struct {
  uint32_t abi_major;
  uint32_t abi_minor;
  uint32_t target_id;
  uint32_t max_command_buffer_bytes;
  uint32_t max_descriptors;
  uint32_t supported_dtype_mask;
  uint32_t native_tile_m;
  uint32_t native_tile_n;
  uint32_t native_tile_k;
  uint64_t sram_bytes;
} NpuCapability;

NpuStatus npuInit(uint32_t requested_abi_major, uint32_t requested_abi_minor);
NpuStatus npuCreateContext(NpuContext **out_context);
NpuStatus npuDestroyContext(NpuContext *context);
NpuStatus npuQueryCapability(NpuContext *context, NpuCapability *out_capability);
NpuStatus npuLoadModule(NpuContext *context, const void *artifact, size_t size, NpuModule **out_module);
NpuStatus npuUnloadModule(NpuModule *module);
NpuStatus npuSubmit(NpuQueue *queue, NpuModule *module, uint32_t command_buffer_index, NpuFence **out_fence);
NpuStatus npuWait(NpuFence *fence, uint64_t timeout_ns);
const char *npuGetStatusString(NpuStatus status);

#ifdef __cplusplus
}
#endif

#endif
