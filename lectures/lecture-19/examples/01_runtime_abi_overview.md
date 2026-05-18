# Runtime ABI Overview

The NPU runtime ABI defines the stable boundary between the MLIR compiler backend and the edge runtime/firmware.

## Required schema blocks

- ABI version and target ID
- tensor descriptor schema
- command buffer schema
- relocation table schema
- runtime C API
- capability query schema
- error/status enum
- debug/profiling metadata

## Golden rule

Do not let runtime infer compiler decisions. Encode layout, strides, quantization policy, and dependency tokens explicitly.
